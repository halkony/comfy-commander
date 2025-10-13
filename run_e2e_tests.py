#!/usr/bin/env python3
"""
Script to run end-to-end tests for Comfy Commander.

This script runs the e2e tests that require a running ComfyUI instance
with the workflow converter extension installed.

Usage:
    python run_e2e_tests.py

Requirements:
    - ComfyUI running on localhost:8188
    - Workflow converter extension installed
    - pytest installed
"""

import subprocess
import sys
import requests
import time


def check_comfyui_server():
    """Check if ComfyUI server is running and accessible."""
    try:
        response = requests.get("http://localhost:8188/system_stats", timeout=5)
        if response.status_code == 200:
            print("[OK] ComfyUI server is running and accessible")
            return True
    except requests.RequestException:
        pass
    
    print("[ERROR] ComfyUI server is not accessible at http://localhost:8188")
    print("   Please start ComfyUI and ensure it's running on port 8188")
    return False


def check_workflow_converter():
    """Check if the workflow converter extension is available."""
    try:
        # Try to access the workflow converter endpoint
        response = requests.post(
            "http://localhost:8188/workflow/convert",
            json={"test": "data"},
            timeout=5
        )
        # We expect this to fail with a 400 or similar, but the endpoint should exist
        if response.status_code in [400, 422]:  # Bad request is expected for invalid data
            print("[OK] Workflow converter extension is available")
            return True
    except requests.RequestException as e:
        pass
    
    print("[ERROR] Workflow converter extension is not available")
    print("   Please install the workflow converter extension:")
    print("   cd ComfyUI/custom_nodes")
    print("   git clone https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint")
    print("   Then restart ComfyUI")
    return False


def run_e2e_tests():
    """Run the e2e tests."""
    print("Running end-to-end tests...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tests/e2e_test_local_server.py", 
            "-v", 
            "--tb=short"
        ], check=True)
        
        print("[SUCCESS] All e2e tests passed!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] Some e2e tests failed (exit code: {e.returncode})")
        return False


def main():
    """Main function to run e2e tests with pre-checks."""
    print("Comfy Commander E2E Test Runner")
    print("=" * 40)
    
    # Check prerequisites
    if not check_comfyui_server():
        sys.exit(1)
    
    if not check_workflow_converter():
        sys.exit(1)
    
    print("\nAll prerequisites met. Running e2e tests...")
    print("-" * 40)
    
    # Run the tests
    if run_e2e_tests():
        print("\n[SUCCESS] All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n[FAILED] Some tests failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
