ARG BUILD_ARCH=amd64
FROM ${BUILD_ARCH}/debian:buster-slim
ARG BUILD_ARCH=amd64

# Install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends --yes \
    libngram-tools sox

# Install pre-built phonetisaurus
ADD phonetisaurus-2019-${BUILD_ARCH}.tar.gz /usr/

COPY pyinstaller/dist/* /usr/lib/rhasspyasr_kaldi_hermes/
COPY debian/bin/* /usr/bin/

ENTRYPOINT ["/usr/bin/rhasspy-asr-kaldi-hermes"]
