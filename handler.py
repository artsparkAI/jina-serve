#!/usr/bin/env python3
"""
RunPod Serverless Handler for Jina Embeddings v4 (vLLM Retrieval)

This is the root entry point. It adds src/ to the Python path and
launches the modular handler from src/handler.py.

For deployment, the Dockerfile uses src/handler.py directly.

Get your Jina AI API key for free: https://jina.ai/?sui=apikey
"""

import sys
import os

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
sys.path.insert(0, src_path)

# Change to src directory so relative imports work
os.chdir(src_path)

# Now run the actual handler module
exec(open(os.path.join(src_path, "handler.py")).read())
