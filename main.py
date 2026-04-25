"""
AI Personal Intelligence Copilot
Main Entry Point

Run this file to launch both the FastAPI backend and Streamlit UI.
Or run them independently:
  Backend:  uvicorn backend.api:app --host 0.0.0.0 --port 8000
  Frontend: streamlit run ui/streamlit_app.py
"""

import subprocess
import sys
import os
import time
import signal
from pathlib import Path

# Ensure we're in the project root
os.chdir(Path(__file__).parent)


def start_backend():
    """Start the FastAPI backend server."""
    print("🚀 Starting FastAPI Backend on port 8000…")
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.api:app",
         "--host", "0.0.0.0", "--port", "8000", "--reload"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def start_frontend():
    """Start the Streamlit frontend."""
    print("🌐 Starting Streamlit UI on port 8501…")
    return subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run",
         "ui/streamlit_app.py",
         "--server.port=8501",
         "--server.address=0.0.0.0",
         "--browser.gatherUsageStats=false",
         "--theme.base=dark"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def main():
    print("=" * 60)
    print("  🧠 AI Personal Intelligence Copilot v1.0")
    print("=" * 60)

    # Create required directories
    for d in ["data", "models/hf_cache", "data/faiss_index", "data/chroma_db"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    processes = []

    try:
        # Start backend
        backend_proc = start_backend()
        processes.append(backend_proc)
        time.sleep(2)   # Give backend time to start

        # Start frontend
        frontend_proc = start_frontend()
        processes.append(frontend_proc)

        print("\n✅ Application started!")
        print("   📡 Backend API:  http://localhost:8000")
        print("   🌐 Streamlit UI: http://localhost:8501")
        print("   📚 API Docs:     http://localhost:8000/docs")
        print("\n   Press Ctrl+C to stop.\n")

        # Wait for processes
        frontend_proc.wait()

    except KeyboardInterrupt:
        print("\n⏹  Shutting down…")

    finally:
        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
        print("✅ All processes stopped.")


if __name__ == "__main__":
    # If called without arguments, launch both services
    if len(sys.argv) == 1:
        main()

    # Single service mode
    elif sys.argv[1] == "backend":
        print("Starting backend only…")
        p = start_backend()
        p.wait()

    elif sys.argv[1] == "frontend":
        print("Starting frontend only…")
        p = start_frontend()
        p.wait()

    elif sys.argv[1] == "demo":
        # Demo mode: launch Streamlit directly (no backend needed)
        print("🎭 Demo mode — launching Streamlit directly")
        os.execvp(
            "streamlit",
            ["streamlit", "run", "ui/streamlit_app.py",
             "--server.port=8501",
             "--theme.base=dark"],
        )
