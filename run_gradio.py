#!/usr/bin/env python3
"""
Attenz AI Agent - Gradio Interface Launcher

This script launches the Gradio web interface for the Attenz AI Agent.
Run this from the project root directory.

Usage:
    python run_gradio.py
"""

import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import and run the gradio interface
from src.server.agent.gradio_interface import main

if __name__ == "__main__":
    main()