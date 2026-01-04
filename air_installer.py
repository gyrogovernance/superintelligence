#!/usr/bin/env python3
"""
AIR Console Installer - checks prerequisites and installs dependencies.

Usage:
    python install_console.py
"""

import subprocess
import sys
import shutil
import os
from pathlib import Path

# Get project root
ROOT = Path(__file__).parent.absolute()
UI_DIR = ROOT / "src" / "app" / "console" / "ui"


def check_python():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"Error: Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_node_npm():
    """Check if Node.js and npm are installed. Returns npm path if found."""
    # Try direct PATH check first
    node_path = shutil.which("node")
    npm_path = shutil.which("npm")
    
    # On Windows, check common installation paths
    if sys.platform == "win32" and not node_path:
        common_node_paths = [
            Path("C:/Program Files/nodejs/node.exe"),
            Path("C:/Program Files (x86)/nodejs/node.exe"),
            Path(os.environ.get("ProgramFiles", "")) / "nodejs" / "node.exe",
            Path(os.environ.get("ProgramFiles(x86)", "")) / "nodejs" / "node.exe",
        ]
        for path in common_node_paths:
            if path.exists():
                node_path = str(path)
                # Add to PATH for this session
                node_dir = str(path.parent)
                current_path = os.environ.get("PATH", "")
                if node_dir not in current_path:
                    os.environ["PATH"] = node_dir + os.pathsep + current_path
                break
    
    if sys.platform == "win32" and not npm_path:
        common_npm_paths = [
            Path("C:/Program Files/nodejs/npm.cmd"),
            Path("C:/Program Files (x86)/nodejs/npm.cmd"),
            Path(os.environ.get("ProgramFiles", "")) / "nodejs" / "npm.cmd",
            Path(os.environ.get("ProgramFiles(x86)", "")) / "nodejs" / "npm.cmd",
        ]
        for path in common_npm_paths:
            if path.exists():
                npm_path = str(path)
                npm_dir = str(path.parent)
                current_path = os.environ.get("PATH", "")
                if npm_dir not in current_path:
                    os.environ["PATH"] = npm_dir + os.pathsep + current_path
                break
    
    if not node_path:
        print("✗ Node.js not found in PATH")
        if sys.platform == "win32":
            print("\nIf you just installed Node.js:")
            print("  1. Close and reopen this terminal/command prompt")
            print("  2. Restart your IDE/editor")
            print("  3. Verify Node.js is installed:")
            print("     - Check: C:/Program Files/nodejs/")
            print("     - Or: C:/Program Files (x86)/nodejs/")
            print("\nIf Node.js is not installed:")
            print("  Download from: https://nodejs.org/")
        else:
            print("\nPlease install Node.js from: https://nodejs.org/")
            print("Or use a Node.js version manager like nvm.")
        return None
    
    # Try to run node
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            check=True,
            shell=sys.platform == "win32",
        )
        node_version = result.stdout.strip()
        print(f"✓ Node.js {node_version} found at: {node_path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"✗ Node.js found at {node_path} but version check failed: {e}")
        return None
    
    if not npm_path:
        print("✗ npm not found (should come with Node.js)")
        print(f"  Node.js is at: {node_path}")
        print("  npm should be in the same directory")
        return None
    
    # Try to run npm
    try:
        result = subprocess.run(
            ["npm", "--version"],
            capture_output=True,
            text=True,
            check=True,
            shell=sys.platform == "win32",
        )
        npm_version = result.stdout.strip()
        print(f"✓ npm {npm_version} found at: {npm_path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"✗ npm found at {npm_path} but version check failed: {e}")
        return None
    
    return npm_path


def check_python_deps():
    """Check if Python dependencies are installed."""
    try:
        import uvicorn
        import fastapi
        # Imports successful - dependencies are installed
        del uvicorn, fastapi  # Clean up to satisfy linter
        print("✓ Python dependencies installed")
        return True
    except ImportError as e:
        print(f"✗ Missing Python dependency: {e.name}")
        print("\nInstall dependencies with:")
        print("  pip install -r requirements.txt")
        return False


def install_frontend_deps(npm_path):
    """Install frontend dependencies."""
    if not UI_DIR.exists():
        print(f"Error: UI directory not found: {UI_DIR}")
        return False
    
    print(f"\nInstalling frontend dependencies in {UI_DIR}...")
    try:
        # Use npm_path directly, or "npm" if it's in PATH
        npm_cmd = npm_path if sys.platform == "win32" and Path(npm_path).exists() else "npm"
        subprocess.run(
            [npm_cmd, "install"],
            cwd=UI_DIR,
            check=True,
            shell=sys.platform == "win32",
            env=os.environ,  # Use environment with updated PATH
        )
        print("✓ Frontend dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install frontend dependencies: {e}")
        return False
    except FileNotFoundError:
        print("✗ npm not found - please install Node.js first")
        return False


def main():
    """Main installer."""
    print("AIR Console Installer")
    print("=" * 50)
    print()
    
    # Check prerequisites
    print("Checking prerequisites...")
    print()
    
    if not check_python():
        sys.exit(1)
    print()
    
    npm_path = check_node_npm()
    if not npm_path:
        print()
        print("Please install Node.js and npm, then run this installer again.")
        sys.exit(1)
    print()
    
    if not check_python_deps():
        print()
        sys.exit(1)
    print()
    
    # Install frontend deps (pass npm_path)
    if not install_frontend_deps(npm_path):
        sys.exit(1)
    
    print()
    print("=" * 50)
    print("Installation complete!")
    print()
    print("Run the console with:")
    print("  python run_console.py")
    print()


if __name__ == "__main__":
    main()

