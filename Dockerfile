FROM ubuntu:eoan as build
ARG TARGETPLATFORM
ARG TARGETARCH
ARG TARGETVARIANT

ENV LANG C.UTF-8

# Install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends --yes \
        python3 python3-dev python3-setuptools python3-pip python3-venv \
        build-essential swig

ENV RHASSPY=/usr/lib/rhasspy
ENV RHASSPY_VENV=$RHASSPY/.venv

# Create virtual environment
COPY Makefile requirements.txt VERSION architecture.sh rhasspy_wheels.txt $RHASSPY/
COPY scripts $RHASSPY/scripts/
COPY download $RHASSPY/download/
RUN cd $RHASSPY && make

# Remove unnecessary files
RUN cd ${RHASSPY_VENV}/lib/python3.7/site-packages/rhasspyasr_kaldi && \
    rm -rf kaldi *.so*

# Create directory for pre-built binaries
RUN mkdir -p ${RHASSPY_VENV}/tools

# Phonetisuarus
ADD download/phonetisaurus-2019-${TARGETARCH}${TARGETVARIANT}.tar.gz /phonetisaurus/
RUN cd /phonetisaurus && mv bin/* ${RHASSPY_VENV}/tools/ && mv lib/* ${RHASSPY_VENV}/tools/

# Kaldi
ADD download/kaldi-2020-${TARGETARCH}${TARGETVARIANT}.tar.gz ${RHASSPY_VENV}/tools/

# -----------------------------------------------------------------------------

FROM ubuntu:eoan
ARG TARGETPLATFORM
ARG TARGETARCH
ARG TARGETVARIANT

ENV LANG C.UTF-8

RUN apt-get update && \
    apt-get install --no-install-recommends --yes \
        python3 perl \
        libatlas3-base libgfortran4 \
        libngram-tools \
        sox

ENV RHASSPY=/usr/lib/rhasspy
ENV RHASSPY_VENV=$RHASSPY/.venv

WORKDIR $RHASSPY

COPY --from=build $RHASSPY_VENV $RHASSPY_VENV
COPY rhasspyasr_kaldi_hermes/ $RHASSPY/rhasspyasr_kaldi_hermes/

ENV PATH=$RHASSPY_VENV/tools:$PATH
ENV LD_LIBRARY_PATH=$RHASSPY_VENV/tools:$LD_LIBRARY_PATH
ENV KALDI_DIR=$RHASSPY_VENV/tools/kaldi

ENTRYPOINT [".venv/bin/python3", "-m", "rhasspyasr_kaldi_hermes"]