# hash:sha256:882053d24d5d73dd277e8e9886c74b7e5bde85d1c256cd3579a20c3621f4fb5a
FROM registry.codeocean.com/codeocean/matlab:2026a-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MLM_LICENSE_FILE

COPY postInstall /
RUN /postInstall
