# Build and model-prep environment. Project files are mounted at run time.
FROM ghcr.io/snapdragon-toolchain/arm64-android@sha256:e0c11123e0104737c5c11643ec64933a11431b5fb95aeb54f85eefa757172332
# arm64-android:v0.3 with Ubuntu 24.04, Android NDK r28b and Hexagon SDK 6.4.0.2 (Tools 19.0.04)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates build-essential \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.5.11@sha256:0ac957607303916420297a4c9c213bb33fbd3c888f9cd7f4f7273596ebf42b85 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python UV_NO_CACHE=1
# full resolved set. Direct: numpy sentencepiece pyyaml tqdm matplotlib. The rest are matplotlib's dependencies.
RUN uv venv /opt/venv --python 3.11 && uv pip install --python /opt/venv/bin/python \
    numpy==1.26.4 sentencepiece==0.2.0 pyyaml==6.0.2 tqdm==4.66.5 matplotlib==3.9.2 \
    contourpy==1.3.3 cycler==0.12.1 fonttools==4.68.0 kiwisolver==1.5.1 packaging==26.3 pillow==13.0.0 \
    pyparsing==3.3.3 python-dateutil==4.0.0 tzdata==2026.6
ENV PATH=/opt/venv/bin:$PATH
