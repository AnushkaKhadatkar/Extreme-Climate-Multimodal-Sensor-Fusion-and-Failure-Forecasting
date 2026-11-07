FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="12.0"
# 12.0 = Blackwell / RTX 5090 ONLY (compute capability reported by every
# RTX 50-series desktop card; DGX Spark's GB10 is 12.1 and would need adding
# separately). Was "8.9" (Ada / RTX 4090) on the machine this image was first
# built and verified on; moved to 12.0 because the project's compute moved to
# an RTX 5090 box. Pin explicitly so spconv/pcdet's nvcc build doesn't waste
# time compiling for every architecture it knows about. On a different GPU
# family, change this again (e.g. "8.6" for RTX 30xx/A6000) or the CUDA ops
# won't load. CUDA base image bumped 11.8 -> 12.8 in the same change: CUDA
# 11.8 predates Blackwell and cannot target sm_120 at all (nvcc rejects it
# outright), and PyTorch's own sm_120 kernels only ship in wheels built
# against CUDA 12.8+ (confirmed via PyTorch's own forum guidance; no stable
# cu118/cu124/cu126 wheel carries sm_120 kernels).

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip git ninja-build libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

RUN pip install --no-cache-dir \
    torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128
# cu128, not cu118: see TORCH_CUDA_ARCH_LIST comment above. 2.7.1 is the
# earliest stable (non-nightly) PyTorch series confirmed by PyTorch's own
# maintainers to ship sm_120 (Blackwell) kernels in its official cu128 wheels.

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
# more robust fix, and lets us use the best-supported (newest) numba. Kept
# unpinned-from-this-history on the move to the RTX 5090 (12.0): the same
# "eager signature + newer-than-tested GPU" trap the decorator fix (patch 3
# below) already defends against generically, so no numba change was needed
# for Blackwell specifically -- verified by actually running the AP eval
# path, not assumed.

# spconv/cumm: use the official prebuilt cu120 wheels (spconv-cu120,
# cumm-cu120), not a from-source build. An earlier version of this file built
# both from source under the belief that no prebuilt wheel carried sm_120
# kernels; that belief was wrong (never actually tested the prebuilt wheel,
# only assumed from its package name predating Blackwell) and the source
# build was also broken in practice: `pip install <git-checkout>` for cumm
# does not run its C++ build step at all (that only happens via `pip install
# -e .`, cumm's documented dev workflow, or a real `cumm-cuXXX` wheel build in
# their manylinux docker) so `cumm.core_cc`, the compiled extension spconv's
# `__init__` imports unconditionally, never existed -- confirmed first-hand:
# `ModuleNotFoundError: No module named 'cumm.core_cc'` on every import.
# The prebuilt spconv-cu120==2.3.6 / cumm-cu120==0.4.11 wheels were tested
# directly against this image's RTX 5090 (compute capability 12.0): imported
# cleanly and ran a real SubMConv3d forward pass on a CUDA tensor with no
# errors. cu120, not cu128, because that's the newest CUDA line spconv
# publishes wheels for; it is forward-compatible with this image's CUDA 12.8
# toolchain for a pip-installed wheel (no compilation against the system CUDA
# happens at install time for a prebuilt wheel).
RUN pip install --no-cache-dir cumm-cu120==0.4.11 spconv-cu120==2.3.6

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
