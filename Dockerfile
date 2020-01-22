ARG BUILD_ARCH=amd64
FROM ${BUILD_ARCH}/debian:buster-slim
ARG BUILD_ARCH=amd64

# Install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends --yes \
    libgfortran3 sox

# Install pre-built mitlm
ADD mitlm-0.4.2-${BUILD_ARCH}.tar.gz /
RUN mv /mitlm/bin/* /usr/bin/
RUN mv /mitlm/lib/* /usr/lib/

# Install pre-built phonetisaurus
ADD phonetisaurus-2019-${BUILD_ARCH}.tar.gz /usr/

COPY pyinstaller/dist/* /usr/lib/rhasspyasr_kaldi_hermes/
COPY debian/bin/* /usr/bin/

ENTRYPOINT ["/usr/bin/rhasspy-asr-kaldi-hermes"]
