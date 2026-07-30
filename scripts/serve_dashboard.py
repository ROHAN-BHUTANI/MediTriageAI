"""Serve the MediTriageAI dashboard and API endpoints."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rich.console import Console
from rich.panel import Panel

console = Console()
DASHBOARD_DIR = REPO_ROOT / "dashboard_web"
RESULTS_DIR = REPO_ROOT / "results"


class MediTriageHandler(BaseHTTPRequestHandler):
    """Custom handler to serve dashboard and API endpoints."""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/":
            self.serve_file("/index.html")
            return
        elif path == "/results":
            self.serve_results()
            return
        elif path.startswith("/static/"):
            # Serve static files from dashboard_web
            self.serve_file(path)
            return
        else:
            # For any other GET, serve from dashboard_web (allows direct access to css/js)
            self.serve_file(path)
            return

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/infer":
            self.handle_infer()
        elif path == "/run":
            self.handle_run_experiment()
        else:
            self.send_error(404, "Endpoint not found")

    def serve_file(self, rel_path: str):
        """Serve a static file from dashboard_web directory."""
        file_path = DASHBOARD_DIR / rel_path.lstrip("/")
        if not file_path.is_file():
            self.send_error(404, "File not found")
            return
        try:
            content = file_path.read_bytes()
            self.send_response(200)
            # Guess content type
            if rel_path.endswith(".html"):
                self.send_header("Content-Type", "text/html")
            elif rel_path.endswith(".css"):
                self.send_header("Content-Type", "text/css")
            elif rel_path.endswith(".js"):
                self.send_header("Content-Type", "application/javascript")
            elif rel_path.endswith(".json"):
                self.send_header("Content-Type", "application/json")
            elif rel_path.endswith(".png"):
                self.send_header("Content-Type", "image/png")
            elif rel_path.endswith(".jpg") or rel_path.endswith(".jpeg"):
                self.send_header("Content-Type", "image/jpeg")
            elif rel_path.endswith(".svg"):
                self.send_header("Content-Type", "image/svg+xml")
            else:
                self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, str(e))

    def serve_results(self):
        """Return current results.json."""
        results_path = DASHBOARD_DIR / "data" / "results.json"
        if results_path.is_file():
            self.serve_file("/data/results.json")
        else:
            self.send_error(404, "Results not found")

    def handle_infer(self):
        """Run infer.py and return JSON."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self.send_error(400, "No input provided")
            return
        post_data = self.rfile.read(content_length)
        try:
            data = json.loads(post_data.decode("utf-8"))
            text = data.get("text", "")
            model = data.get("model", "xlm_roberta")
        except (json.JSONDecodeError, KeyError):
            self.send_error(400, "Invalid JSON")
            return

        if not text:
            self.send_error(400, "Missing 'text' field")
            return

        # Run infer.py as subprocess
        cmd = [
            sys.executable,
            str(REPO_ROOT / "scripts" / "infer.py"),
            "--model",
            model,
            "--text",
            text,
            "--output-format",
            "json",
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                console.print(f"[red]Inference failed: {result.stderr}[/red]")
                self.send_error(500, "Inference failed")
                return
            output = json.loads(result.stdout)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(output).encode("utf-8"))
        except subprocess.TimeoutExpired:
            self.send_error(504, "Inference timeout")
        except Exception as e:
            console.print(f"[red]Inference error: {e}[/red]")
            self.send_error(500, "Inference error")

    def handle_run_experiment(self):
        """Trigger run_experiment.py in background."""

        # Run in a separate thread so we don't block the HTTP request
        def run_async():
            try:
                subprocess.run(
                    [sys.executable, str(REPO_ROOT / "scripts" / "run_experiment.py")],
                    check=False,
                )
            except Exception as e:
                console.print(f"[red]Experiment error: {e}[/red]")

        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()

        self.send_response(202)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "started"}).encode("utf-8"))

    def log_message(self, format, *args):
        """Override to use rich console for logging."""
        console.log(f"{self.address_string()} - {format % args}")


def run_server(port: int = 8080):
    """Start the HTTP server."""
    server_address = ("", port)
    httpd = ThreadingHTTPServer(server_address, MediTriageHandler)
    url = f"http://localhost:{port}"
    console.print(
        Panel.fit(
            f"[bold]MediTriageAI Dashboard[/bold]\n{url}\nPress Ctrl+C to stop",
            border_style="blue",
        )
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down server...[/yellow]")
        httpd.shutdown()


if __name__ == "__main__":
    run_server()
