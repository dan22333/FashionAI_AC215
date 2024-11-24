# Use Python base image
FROM python:3.9-slim

# Set default environment variables for build-time configuration
ARG DEBIAN_PACKAGES="build-essential git screen vim"
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV PYENV_SHELL=/bin/bash
ENV PYTHONUNBUFFERED=1

# Install system dependencies and Python tools
RUN set -ex; \
    for i in $(seq 1 8); do mkdir -p "/usr/share/man/man${i}"; done && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends $DEBIAN_PACKAGES curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip && \
    pip install pipenv && \
    useradd -ms /bin/bash app -d /home/app -u 1000 -p "$(openssl passwd -1 Passw0rd)" && \
    mkdir -p /app && \
    chown app:app /app


# Set the working directory
WORKDIR /app

# Copy Pipfile and Pipfile.lock for dependency installation
COPY Pipfile Pipfile.lock ./

# Install dependencies using Pipenv
RUN pipenv install --deploy --ignore-pipfile

# Copy the application code
COPY . .


# Use environment variables to dynamically set the host and port for Uvicorn
CMD ["sh", "-c", "pipenv run uvicorn main:app --host ${APP_HOST:-0.0.0.0} --port ${APP_PORT_VECTOR:-8001}"]
