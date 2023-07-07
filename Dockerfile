ARG DEBIAN_VERSION=bullseye-slim

FROM public.ecr.aws/debian/debian:$DEBIAN_VERSION

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    dh-autoreconf \
    python3-networkx \
    libhdf5-dev \
    libgtk-3-dev \
    git \
    && apt-get clean -y \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

COPY ./ /tmp

WORKDIR /tmp

RUN autoreconf \
    && ./configure --prefix=/opt/MARVEL \
    && make \
    && make install \
    && rm -rf /tmp/*

ENV PYTHONPATH="/opt/MARVEL/lib.python:$PYTHONPATH"
ENV PATH="/opt/MARVEL/bin:/opt/MARVEL/scripts:$PATH"