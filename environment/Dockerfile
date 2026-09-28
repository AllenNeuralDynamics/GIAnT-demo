# hash:sha256:be3d713439cd0ae67af35f2582d75ad18cf62a9c5e4d40eb4dc5eb69f57fecfb
FROM registry.codeocean.com/codeocean/matlab:2026a-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MLM_LICENSE_FILE

RUN pip install -U --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    aind-ophys-utils \
    matplotlib \
    opencv-python-headless

COPY postInstall /
RUN /postInstall
