# # Use Python 3.8 slim image
# FROM python:3.8-slim

# # Argument for Debian packages to be installed
# ARG DEBIAN_PACKAGES="build-essential git screen vim"

# # Prevent apt from showing prompts
# ENV DEBIAN_FRONTEND=noninteractive

# # Set UTF-8 locale for Python
# ENV LANG=C.UTF-8

# # Tell Python to disable buffering for logs
# ENV PYTHONUNBUFFERED=1

# # Ensure up-to-date baseline, install dependencies, and create user
# RUN set -ex; \
#     for i in $(seq 1 8); do mkdir -p "/usr/share/man/man${i}"; done && \
#     apt-get update && \
#     apt-get upgrade -y && \
#     apt-get install -y --no-install-recommends $DEBIAN_PACKAGES && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/* && \
#     pip install --no-cache-dir --upgrade pip && \
#     useradd -ms /bin/bash app -d /home/app -u 1000 -p "$(openssl passwd -1 Passw0rd)" && \
#     mkdir -p /app && \
#     mkdir -p /persistent && \
#     chown app:app /persistent && \
#     chown app:app /app

# # Create the /src directory inside the container
# RUN mkdir -p /src

# # Switch to the new user
# USER app

# # Set PATH to include the local bin directory
# ENV PATH=$PATH:/home/app/.local/bin

# # Set the working directory to /app
# WORKDIR /app

# # Install pipenv
# RUN pip install pipenv

# # # Copy the Python scripts (finetune.py and qa.py) into the container
# # COPY --chown=app:app finetune.py qa.py /app/

# # # copy everything from the current directory into the container
# # COPY --chown=app:app . /app/

# # Install any required dependencies
# ADD --chown=app:app Pipfile Pipfile.lock /app/
# RUN pipenv sync

# # Set the default working directory for Python execution
# WORKDIR /app

# # # Use ENTRYPOINT to open bash and automatically start pipenv shell
# # ENTRYPOINT ["/bin/bash", "-c", "pipenv shell && exec \"$@\"", "--"]

# CMD ["pipenv", "run", "python", "/src/finetune.py"]




# Use the official Ubuntu image as the base
FROM ubuntu:20.04

# Set the environment variable for non-interactive installations
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV PYENV_SHELL=/bin/bash
ENV PYTHONUNBUFFERED=1

# Install required dependencies
RUN apt-get update && \
    apt-get install -y curl apt-transport-https ca-certificates gnupg lsb-release openssh-client

# Add the Google Cloud SDK repository
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg

# Docker
RUN install -m 0755 -d /etc/apt/keyrings
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
RUN chmod a+r /etc/apt/keyrings/docker.gpg
RUN echo "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install packages
RUN apt-get update && \
    apt-get install -y google-cloud-sdk google-cloud-sdk-gke-gcloud-auth-plugin jq docker-ce

# Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.9 python3-pip && \
    pip install pipenv

RUN useradd -ms /bin/bash app -d /home/app -u 1000 -p "$(openssl passwd -1 passw0rd)" && \
    usermod -aG docker app && \
    mkdir -p /app && \
    chown app:app /app

# Set the working directory
WORKDIR /app

ADD --chown=app:app . /app

# Start a new shell to use the installed SDK
ENTRYPOINT ["/bin/bash","./docker-entrypoint.sh"]