# Use the official Ubuntu 22.04 LTS as a base image
FROM ubuntu:22.04

# Set non-interactive mode for apt-get
ARG DEBIAN_FRONTEND=noninteractive

# Update the package list and install prerequisites
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y software-properties-common git && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default Python version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create a working directory named 'rendernet'
WORKDIR /rendernet

# Clone the SAM2 repository into the working directory
RUN git clone https://github.com/roverxio/SAM2 .

# Create a virtual environment and activate it
RUN python3 -m venv venv

# Install the package in editable mode
RUN . venv/bin/activate && pip install -e .

# Install dependencies from requirements.txt
RUN . venv/bin/activate && pip install -r requirements.txt

# Download SAM checkpoint
RUN cd checkpoints && ./download_ckpts.sh

# Copy config file
Copy config.yaml .

# Set environment variables (modify these as needed)
ENV AWS_ACCESS_KEY_ID=access_key
ENV AWS_SECRET_ACCESS_KEY=secret_key
ENV AWS_REGION=region

EXPOSE 8080

# Command to run when starting the container (modify as needed)
CMD ["bash", "-c", "source venv/bin/activate && python3 main.py"]