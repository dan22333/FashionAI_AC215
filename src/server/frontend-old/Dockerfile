# Use Node.js base image
FROM node:14.9.0-buster-slim

# Update and install http-server
RUN apt-get update && \
    apt-get install -y --no-install-recommends openssl && \
    npm install -g http-server

# Set working directory and copy files
WORKDIR /app
COPY . /app

# Expose port for http-server
EXPOSE 8080

# Run http-server on port 8080
CMD ["http-server", "/app", "-p", "8080"]
