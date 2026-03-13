"""
Project: Microclimate Forecasting System using Machine Learning
Purpose: Serve a human-friendly frontend dashboard for forecast/control outputs.
Module Group: Visualization & Reporting Module (UI delivery)
DFD Connection: Presents ML forecasts and decision-simulation states to end users.
"""

from __future__ import annotations

from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

import pandas as pd

from src.config import PROJECT_ROOT, REPORTS_DIR
from src.config import DEFAULT_HIGH_THRESHOLD_C, DEFAULT_LOW_THRESHOLD_C, DEFAULT_SPRAY_THRESHOLD_C
from src.pipeline_data_prep import run_full_project_workflow

FRONTEND_DIR = PROJECT_ROOT / "frontend"
DASHBOARD_INDEX_PATH = REPORTS_DIR / "ui_payload" / "dashboard_payload.json"
OVERALL_RESULTS_PATH = REPORTS_DIR / "overall_crop_results.csv"
PLANT_SUMMARY_PATH = REPORTS_DIR / "plant_growth_intelligence" / "plant_growth_intelligence_summary.json"


def _as_number(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _load_plant_intelligence_fallback() -> Dict[str, Any]:
    if not PLANT_SUMMARY_PATH.exists():
        return {}
    try:
        payload = json.loads(PLANT_SUMMARY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_dashboard_bundle() -> Dict[str, Any]:
    if not DASHBOARD_INDEX_PATH.exists():
        run_full_project_workflow()

    index_payload = json.loads(DASHBOARD_INDEX_PATH.read_text(encoding="utf-8"))

    metrics_by_crop: Dict[str, Dict[str, Any]] = {}
    if OVERALL_RESULTS_PATH.exists():
        overall_df = pd.read_csv(OVERALL_RESULTS_PATH)
        for _, row in overall_df.iterrows():
            crop_name = str(row.get("crop_type", "")).lower()
            metrics_by_crop[crop_name] = {
                "mae": _as_number(row.get("mae")),
                "rmse": _as_number(row.get("rmse")),
                "mape_pct": _as_number(row.get("mape_pct")),
                "r2": _as_number(row.get("r2")),
                "samples": int(row.get("sample_count", 0)),
                "model_coordination": str(row.get("model_coordination", "unknown")),
                "best_model": str(row.get("best_model", "unknown")),
                "best_model_rmse": _as_number(row.get("best_model_rmse")),
            }

    crop_entries: List[Dict[str, Any]] = []
    for item in index_payload.get("payloads", []):
        payload_path = Path(str(item.get("payload_file", "")))
        if not payload_path.exists():
            continue

        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        crop_key = str(item.get("crop_type", payload.get("crop_type", "unknown"))).lower()
        crop_entries.append(
            {
                "crop_type": crop_key.upper(),
                "raw_crop_type": crop_key,
                "model_coordination": str(item.get("model_coordination", "unknown")),
                "timestamps": payload.get("timestamps", []),
                "actual_temperature_c": payload.get("actual_temperature_c", []),
                "predicted_temperature_c": payload.get("predicted_temperature_c", []),
                "absolute_error_c": payload.get("absolute_error_c", []),
                "actions": payload.get("actions", []),
                "fan_on": payload.get("fan_on", []),
                "spray_on": payload.get("spray_on", []),
                "model_weights": payload.get("model_weights", {}),
                "validation_rmse": payload.get("validation_rmse", {}),
                "model_comparison": payload.get("model_comparison", []),
                "best_model": payload.get("best_model"),
                "model_prediction_series": payload.get("model_prediction_series", {}),
                "metrics": metrics_by_crop.get(crop_key, {}),
            }
        )

    plant_info_raw = index_payload.get("plant_growth_intelligence", {})
    plant_payload: Dict[str, Any] = plant_info_raw if isinstance(plant_info_raw, dict) else {}
    if not plant_payload or plant_payload.get("status") != "completed":
        fallback_payload = _load_plant_intelligence_fallback()
        if fallback_payload:
            plant_payload = fallback_payload

    figure_paths = plant_payload.get("figures", {}) if isinstance(plant_payload.get("figures", {}), dict) else {}
    figure_api_urls = {}
    for key, raw_path in figure_paths.items():
        path_obj = Path(str(raw_path))
        if path_obj.exists():
            figure_api_urls[str(key)] = f"/api/plant-figure?key={key}"
    plant_payload["figure_api_urls"] = figure_api_urls

    crop_entries.sort(key=lambda entry: entry["crop_type"])
    return {
        "project": index_payload.get("project", "Microclimate Forecasting System"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "crops_processed": int(index_payload.get("crops_processed", len(crop_entries))),
        "thresholds": {
            "low_threshold_c": DEFAULT_LOW_THRESHOLD_C,
            "high_threshold_c": DEFAULT_HIGH_THRESHOLD_C,
            "spray_threshold_c": DEFAULT_SPRAY_THRESHOLD_C,
        },
        "plant_growth_intelligence": plant_payload,
        "crops": crop_entries,
    }


def _resolve_plant_figure_path(figure_key: str) -> Path | None:
    plant_info: Dict[str, Any] = {}
    if DASHBOARD_INDEX_PATH.exists():
        index_payload = json.loads(DASHBOARD_INDEX_PATH.read_text(encoding="utf-8"))
        raw = index_payload.get("plant_growth_intelligence", {})
        if isinstance(raw, dict):
            plant_info = raw
    if not plant_info or plant_info.get("status") != "completed":
        fallback = _load_plant_intelligence_fallback()
        if isinstance(fallback, dict) and fallback:
            plant_info = fallback

    if not isinstance(plant_info, dict):
        return None

    figures = plant_info.get("figures", {})
    if not isinstance(figures, dict):
        return None

    raw_path = figures.get(figure_key)
    if raw_path is None:
        return None

    figure_path = Path(str(raw_path)).resolve()
    project_root = PROJECT_ROOT.resolve()
    try:
        figure_path.relative_to(project_root)
    except Exception:
        return None

    if not figure_path.exists() or not figure_path.is_file():
        return None
    return figure_path


class DashboardRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        # Disable browser caching for static assets during active UI iteration.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def _send_json(self, payload: Dict[str, Any], status_code: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_binary_file(self, file_path: Path, status_code: int = 200) -> None:
        content = file_path.read_bytes()
        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/plant-figure":
            params = parse_qs(parsed.query)
            figure_key = str(params.get("key", [""])[0]).strip()
            if not figure_key:
                self._send_json({"error": "Missing figure key."}, status_code=400)
                return
            figure_path = _resolve_plant_figure_path(figure_key)
            if figure_path is None:
                self._send_json({"error": f"Unknown or unavailable figure key '{figure_key}'."}, status_code=404)
                return
            self._send_binary_file(figure_path, status_code=200)
            return

        if parsed.path == "/api/dashboard":
            params = parse_qs(parsed.query)
            refresh_flag = params.get("refresh", ["0"])[0] == "1"
            if refresh_flag:
                try:
                    run_full_project_workflow()
                except Exception as exc:
                    self._send_json({"error": f"Pipeline refresh failed: {exc}"}, status_code=500)
                    return
            try:
                payload = _load_dashboard_bundle()
            except Exception as exc:
                self._send_json({"error": f"Dashboard load failed: {exc}"}, status_code=500)
                return
            self._send_json(payload)
            return

        if parsed.path == "/api/refresh":
            try:
                run_full_project_workflow()
                payload = _load_dashboard_bundle()
            except Exception as exc:
                self._send_json({"error": f"Refresh failed: {exc}"}, status_code=500)
                return
            self._send_json(payload)
            return

        if parsed.path == "/":
            self.path = "/index.html"
        super().do_GET()


def run_dashboard_server(host: str = "127.0.0.1", port: int = 8080, refresh_before_start: bool = True) -> None:
    FRONTEND_DIR.mkdir(parents=True, exist_ok=True)
    if refresh_before_start:
        run_full_project_workflow()

    handler = partial(DashboardRequestHandler, directory=str(FRONTEND_DIR))
    server = ThreadingHTTPServer((host, port), handler)

    print(f"Dashboard URL: http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frontend dashboard for greenhouse forecast outputs.")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port number (default: 8080)")
    parser.add_argument(
        "--no-refresh",
        action="store_true",
        help="Skip pipeline refresh before starting server.",
    )
    args = parser.parse_args()

    run_dashboard_server(
        host=args.host,
        port=args.port,
        refresh_before_start=not args.no_refresh,
    )


if __name__ == "__main__":
    main()

