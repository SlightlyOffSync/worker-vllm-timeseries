FROM nvidia/cuda:12.1.0-base-ubuntu22.04

RUN apt-get update -y && \
    # Install prerequisites for adding new repositories and for Python
    apt-get install -y software-properties-common && \
    # Add the deadsnakes PPA for newer Python versions
    add-apt-repository ppa:deadsnakes/ppa && \
    # Install Python 3.11 and its development headers
    apt-get install -y python3.11 python3.11-dev python3.11-distutils git && \
    # Install pip for Python 3.11
    apt-get install -y python3-pip && \
    # Clean up apt cache
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Use update-alternatives to make python3.11 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2

# Verify the new default version
RUN python3 --version

RUN ldconfig /usr/local/cuda-12.1/compat/

# Install Python dependencies
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade -r /requirements.txt

# --- START OF MODIFIED BLOCK ---
# Clone the vLLM fork first, then install it with the precompiled flag.
# This ensures build dependencies are handled correctly.
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --branch timeseries https://github.com/xiez22/vllm.git /vllm-workspace && \
    python3 -m pip install /vllm-workspace && \
    python3 -m pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3
# --- END OF MODIFIED BLOCK ---

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0

# This PYTHONPATH now correctly points to the cloned repository
ENV PYTHONPATH="/:/vllm-workspace"


COPY src /src
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python3 /src/download_model.py; \
    fi

# Start the handler
CMD ["python3", "/src/handler.py"]
