FROM docker.io/nvidia/cuda:12.3.2-runtime-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "transformers<5" \
    whisperx \
    pyannote.audio \
    runpod \
    && pip cache purge

# Pre-download gated diarization models using build secret
RUN --mount=type=secret,id=hf_token \
    python3 -c "
import os
token = open('/run/secrets/hf_token').read().strip()
from huggingface_hub import snapshot_download
snapshot_download('pyannote/speaker-diarization-3.1', token=token)
snapshot_download('pyannote/segmentation-3.0', token=token)
print('Models cached')
"

COPY src/handler.py /handler.py

RUN python3 -c "import whisperx; import pyannote.audio; import runpod; print('All imports OK')"

CMD ["python3", "-u", "/handler.py"]
