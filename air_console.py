#!/usr/bin/env python3
"""
Run AIR Console - starts both backend and frontend servers.

Usage:
    python run_console.py

Press Ctrl+C to stop both servers.
"""

import subprocess
import sys
import signal
import time
import shutil
import os
import types
from pathlib import Path
from typing import List, NoReturn

# Get project root
ROOT = Path(__file__).parent.absolute()
UI_DIR = ROOT / "src" / "app" / "console" / "ui"

# Store process references
processes: List[subprocess.Popen[bytes]] = []


def cleanup():
    """Terminate all subprocesses."""
    print("\nShutting down servers...")
    for proc in processes:
        if proc.poll() is None:  # Still running
            try:
                if sys.platform == "win32":
                    # On Windows, use kill() directly for more reliable shutdown
                    proc.kill()
                else:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            except ProcessLookupError:
                # Process already terminated
                pass
    # Give processes a moment to fully exit
    time.sleep(0.5)
    print("Servers stopped.")


def signal_handler(sig: int, frame: types.FrameType | None) -> NoReturn:
    """Handle Ctrl+C gracefully."""
    cleanup()
    sys.exit(0)


def run_backend():
    """Run FastAPI backend server."""
    print("Starting backend server on http://localhost:8000...")
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.app.console.api.server:app",
            "--reload",
            "--port",
            "8000",
        ],
        cwd=ROOT,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )


def run_frontend():
    """Run Vite frontend dev server."""
    print("Starting frontend dev server on http://localhost:5173...")
    
    # Check npm is available
    npm = shutil.which("npm")
    
    # On Windows, check common installation paths
    if sys.platform == "win32" and not npm:
        common_npm_paths = [
            Path("C:/Program Files/nodejs/npm.cmd"),
            Path("C:/Program Files (x86)/nodejs/npm.cmd"),
            Path(os.environ.get("ProgramFiles", "")) / "nodejs" / "npm.cmd",
            Path(os.environ.get("ProgramFiles(x86)", "")) / "nodejs" / "npm.cmd",
        ]
        for path in common_npm_paths:
            if path.exists():
                npm = str(path)
                # Add to PATH for this session
                npm_dir = str(path.parent)
                current_path = os.environ.get("PATH", "")
                if npm_dir not in current_path:
                    os.environ["PATH"] = npm_dir + os.pathsep + current_path
                break
    
    if not npm:
        raise FileNotFoundError(
            "npm not found. Please run 'python install_console.py' first."
        )
    
    # Check node_modules exists
    if not (UI_DIR / "node_modules").exists():
        raise FileNotFoundError(
            f"node_modules not found in {UI_DIR}. "
            "Please run 'python install_console.py' first."
        )

    # On Windows, use creationflags to create a new process group for better cleanup
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP

    return subprocess.Popen(
        [npm, "run", "dev"],
        cwd=UI_DIR,
        stdout=sys.stdout,
        stderr=sys.stderr,
        shell=sys.platform == "win32",
        env=os.environ,  # Use environment with updated PATH
        creationflags=creation_flags if sys.platform == "win32" else 0,
    )


def main():
    """Main entry point."""
    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Check UI directory exists
    if not UI_DIR.exists():
        print(f"Error: UI directory not found: {UI_DIR}")
        sys.exit(1)

    print("AIR Console Runner")
    print("=" * 50)
    print("Backend:  http://localhost:8000")
    print("Frontend: http://localhost:5173")
    print("Press Ctrl+C to stop both servers")
    print("=" * 50)
    print()

    try:
        # Start backend
        backend_proc = run_backend()
        processes.append(backend_proc)

        # Give backend a moment to start
        time.sleep(1)

        # Start frontend
        frontend_proc = run_frontend()
        processes.append(frontend_proc)

        # Wait for either process to exit
        try:
            while True:
                if backend_proc.poll() is not None:
                    print("Backend server exited unexpectedly.")
                    break
                if frontend_proc.poll() is not None:
                    print("Frontend server exited unexpectedly.")
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            # Signal handler will call cleanup()
            pass

    except KeyboardInterrupt:
        # Fallback if signal handler didn't work
        cleanup()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        cleanup()


if __name__ == "__main__":
    main()

