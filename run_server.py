"""Launcher for Push server to resolve relative imports."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uvicorn
from server.config import SERVER_HOST, SERVER_PORT
uvicorn.run("server.main:app", host=SERVER_HOST, port=SERVER_PORT, log_level="info")
