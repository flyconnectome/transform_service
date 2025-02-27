# Coordinate Query & Conversion Service

This code is based on [CloudVolumeServer](https://github.com/flyconnectome/CloudVolumeServer).

## Requirements

This project uses [`uv`](https://github.com/astral-sh/uv) to manage dependencies and the virtual environment.

```bash
# Run to setup dependencies
uv sync
```

If you are running this serve in production, you probably want a virtual environment:

```bash
# Create a virtual environment
uv env
```

## Run the web service locally
```uv run uvicorn --reload app.main:app```

## Run tests
```uv run pytest```

## Run in production

We deployed this service using:

- `gunicorn` as the webserver
- `supervisor` as the process control system for the gunicorn server
- a reverse proxy set up in `nginx` forwarding requests to the gunicorn webserver

Please see [this tutorial](https://dylancastillo.co/posts/fastapi-nginx-gunicorn.html) for step-by-step instructions.
