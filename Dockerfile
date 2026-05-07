FROM docker.io/nvidia/cuda:12.3.2-runtime-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Don't pre-install torch — pip resolves it from whisperx's requirements
# PyPI's default CUDA builds (12.x) match the 12.3.2 runtime
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "transformers<5" \
    whisperx \
    pyannote.audio \
    runpod \
    && pip cache purge

COPY src/handler.py /handler.py

# Verify
RUN python3 -c "import whisperx; import pyannote.audio; import runpod; print('All imports OK')"

CMD ["python3", "-u", "/handler.py"]
