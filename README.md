# Connectome Services

This repository contains code for running various connectome-related services related currently deployed
on `flyem.mrc-lmb.cam.ac.uk` (internally `flyem1`):

- a supervoxel look-up service for the Zheng et al. CA3 volume
- a service for dynamically generating neuroglancer segmentation properties from tables on flytable

## Requirements

The project uses [`uv`](https://github.com/astral-sh/uv) to manage dependencies and the virtual environment.

```bash
# Run to setup dependencies
uv sync
```

If you are running this server in production, you probably want a virtual environment:

```bash
# Create a virtual environment
uv env
```

In order to use the segment property service, you will also need to setup environment variables:
- `SEATABLE_SERVER`: URL for FlyTable
- `SEATABLE_TOKEN`: API token for FlyTable

_Note to self: on `flyem1` I added these to the `gunicorn_start` script._

## Run the web service locally
```uv run uvicorn --reload app.main:app```

## Run tests
```uv run pytest```

## Run in production

We deployed this service on `flyem1` using:

- `gunicorn` as the webserver
- `supervisor` as the process control system for the gunicorn server
- a reverse proxy set up in `nginx` forwarding requests to the gunicorn webserver

Please see [this tutorial](https://dylancastillo.co/posts/fastapi-nginx-gunicorn.html) for general step-by-step instructions.

[This Slack message](https://flyconnectome.slack.com/archives/C29G9694H/p1740648367149559) contains details on how
the service is currently deployed on `flyem1`.

To restart the service (e.g. after changing the code), you can run:

```bash
sudo supervisorctl restart transform_service
```

