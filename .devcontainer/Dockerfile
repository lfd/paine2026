# Basis-Image mit variabler Python-Version
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}

# Arbeitsverzeichnis
WORKDIR /workspace

# System-Tools installieren (optional)
RUN apt-get update && apt-get install -y \
    curl git && \
    rm -rf /var/lib/apt/lists/*

# Jupyter vorbereiten
RUN pip install --upgrade pip && \
    pip install jupyter

# Non-root User für VS Code
RUN useradd -ms /bin/bash vscode
USER vscode

# Standard-Port für Jupyter
EXPOSE 8883