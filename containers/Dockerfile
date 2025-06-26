FROM python:3.11

WORKDIR /app
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git

RUN python -m venv venv
RUN git init
RUN venv/bin/pip install .

# Execute the application
CMD ["/app/venv/bin/python", "/app/example_applications/simple_web_viewer.py"]