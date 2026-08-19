
uv venv
uv sync

clone dsa && cd dsa
UV_HTTP_TIMEOUT=300 uv pip install -e .[dev] --python ../.venv/bin/python
