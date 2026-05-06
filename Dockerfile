FROM docker.io/runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Cleanup to free space
RUN pip cache purge && rm -rf /root/.cache/pip /root/.cache/torch && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# CRITICAL: Pin torchaudio to match the base image's torch 2.1.0+cu118
# Without this, pip pulls torchaudio for CUDA 12.x which ABI-mismatches
RUN pip install --no-cache-dir \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118 \
    --force-reinstall

# Now install whisperx + pyannote (torch won't be touched since it's already correct)
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "transformers<4.36" \
    whisperx \
    pyannote.audio \
    runpod \
    && pip cache purge

# Pre-cache base.en model and alignment model
RUN python3 -c "import whisperx; whisperx.load_model('base.en', 'cpu', compute_type='float32')" 2>/dev/null || true
RUN python3 -c "import whisperx; whisperx.load_align_model(language_code='en', device='cpu')" 2>/dev/null || true

COPY src/handler.py /handler.py

# Verify imports at build time
RUN python3 -c "import whisperx; import pyannote.audio; import runpod; print('All imports OK')"

CMD ["python3", "-u", "/handler.py"]
