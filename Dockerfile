FROM runpod/base:0.7.4-cuda12.1.0

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages (pin to avoid conflicts)
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "torch==2.1.0" \
    "torchaudio==2.1.0" \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    "transformers<4.36" \
    whisperx \
    pyannote.audio \
    runpod

# Pre-download Whisper base.en model
RUN python3 -c "import whisperx; whisperx.load_model('base.en', 'cpu', compute_type='float32')" 2>/dev/null

# Pre-download alignment model
RUN python3 -c "import whisperx; whisperx.load_align_model(language_code='en', device='cpu')" 2>/dev/null

# Pre-download PyAnnote models (requires auth at runtime, so we cache the base)
# These download on first cold start with a token

COPY src/handler.py /handler.py
RUN chmod +x /handler.py

CMD ["python3", "-u", "/handler.py"]
