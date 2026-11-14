# Model 1 image: the baseline's pinned OpenPCDet image plus what Model 1 adds.
# Build the base first (baseline/docker/, tag stf-openpcdet:pinned), then:
#   docker build -t stf-model1:pinned models/model1/docker
#
# Kept as a thin layer on top of the baseline image on purpose: same CUDA,
# PyTorch, spconv and OpenPCDet commit as the baseline, so Model 1 and the
# baseline are trained and evaluated by the same code (context.md section 6).
FROM stf-openpcdet:pinned

# pytest for models/model1/tests. kornia: OpenPCDet's load_data_to_gpu converts
# the "images" batch key with kornia.image_to_tensor, and the baseline image (no
# images) doesn't ship it. 0.6.12 supports torch 2.0. Pinned like the base image.
RUN pip install --no-cache-dir pytest==8.3.5 kornia==0.6.12

# PyTorch >= 2.6 defaults torch.load to weights_only=True, which rejects OpenPCDet's own
# checkpoints (they contain numpy scalars): every checkpoint load, i.e. the evaluation after
# training and test.py --ckpt/--eval_all, crashed with "UnpicklingError: Weights only load
# failed" (seen on the RTX 5090 image, torch 2.7.1, M3 run on 2026-11-13). weights_only=False
# is safe here because only checkpoints we trained ourselves are loaded. Patched in this layer,
# not in baseline/docker, so the baseline image stays as it is. The grep -c asserts the
# four call sites in detector3d_template.py were all changed, so a changed OpenPCDet commit
# fails the build instead of silently skipping the patch.
RUN sed -i -E 's/torch\.load\((filename|pre_trained_path|optimizer_filename), map_location=loc_type\)/torch.load(\1, map_location=loc_type, weights_only=False)/' \
        /opt/OpenPCDet/pcdet/models/detectors/detector3d_template.py \
    && test "$(grep -c 'weights_only=False' /opt/OpenPCDet/pcdet/models/detectors/detector3d_template.py)" = "4"

# nvJitLink version mismatch (seen on the rebuilt 5090 image, M3 evaluation, 2026-11-13):
# baseline/docker/requirements.lock.txt pins nvidia-nvjitlink-cu12==12.9.86 and installs
# nvidia-cuda-nvcc-cu12==12.9.86, but torch 2.7.1+cu128 itself requires nvjitlink 12.8.61, and
# the image ended up with 12.8.61. numba-cuda then compiles IoU kernels (KITTI AP scoring, the
# rotate_iou step) with the 12.9 NVVM and fails to link: "nvJitLinkError ERROR_INTERNAL ...
# may need newer version of nvJitLink library". Install the pinned 12.9.86 (--no-deps so torch's
# pin does not pull it back down); CUDA minor-version compatibility keeps torch working.
RUN pip install --no-cache-dir --no-deps nvidia-nvjitlink-cu12==12.9.86

# Bake the ImageNet ResNet-50 weights into the image so training never depends
# on a download at run time (and every run uses the identical weights file).
RUN python -c "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"

WORKDIR /workspace
