FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="8.9"
# 8.9 = Ada / RTX 4090 ONLY. Pin explicitly so spconv/pcdet's nvcc build doesn't
# waste time compiling for every architecture it knows about. On a different GPU
# family, change this (e.g. "8.6" for RTX 30xx/A6000) or the CUDA ops won't load.

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip git ninja-build libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

COPY requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements.lock.txt
# Every package is pinned in requirements.lock.txt (pip freeze of a working
# build), including numba 0.68.1. History of the numba choice, kept for context:
# A version-pin was tried first (0.56.4, to dodge a "Signature mismatch"
# TypeError on latest numba) and made things worse: it traded that error for
# a silent segfault, i.e. this GPU's compute capability (8.9, Ada -- RTX 4090
# launched after numba 0.56 shipped) needs a newer numba codegen than the
# file's eager string-signature decorators were tested against. Fixing the
# decorators themselves (next step) instead of hunting for a numba version
# that happens to tolerate both the old decorator style and this GPU is the
# more robust fix, and lets us use the best-supported (newest) numba.

# Pinned OpenPCDet commit (context.md section 6: pin versions). patch_openpcdet.py
# asserts exact match counts against this tree, so bump this deliberately and
# re-verify the patches, never implicitly by rebuilding.
ARG OPENPCDET_SHA=233f849829b6ac19afb8af8837a0246890908755
WORKDIR /opt/OpenPCDet
RUN git init -q && git remote add origin https://github.com/open-mmlab/OpenPCDet.git \
    && git fetch -q --depth 1 origin ${OPENPCDET_SHA} && git checkout -q FETCH_HEAD
WORKDIR /opt

# Three build-time patches to the OpenPCDet checkout -- see
# patch_openpcdet.py's own comments for why each one is needed (unused
# optional dataset connectors pulling in heavy unrelated devkits, a missing
# pathlib.Path import that's an upstream bug, and numba/CUDA eager-signature
# decorators that crash two different ways on this GPU). A COPY'd script
# instead of a Dockerfile heredoc: an earlier heredoc version of the
# rotate_iou.py regex patch hit a reproducible discrepancy between what ran
# inside `docker build` and the identical regex run manually, so a real file
# sidesteps that whole class of problem.
COPY patch_openpcdet.py /tmp/patch_openpcdet.py
RUN python /tmp/patch_openpcdet.py

WORKDIR /opt/OpenPCDet
RUN python setup.py develop

WORKDIR /workspace
