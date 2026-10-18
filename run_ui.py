import os
import sys
import uvicorn

# Setup paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_APP_DIR = os.path.join(PROJECT_ROOT, "web_app")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if WEB_APP_DIR not in sys.path:
    sys.path.insert(0, WEB_APP_DIR)

from web_app.app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    url = f"http://localhost:{port}"
    print("=" * 65)
    print("  FOCAL: Multimodal Time-Series Sensing Web UI (NotebookLM Style)")
    print(f"  Truy cap giao dien tai: {url}")
    print("  Nhan Ctrl+C de dung server.")
    print("=" * 65)
    
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
