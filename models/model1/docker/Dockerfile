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

# Bake the ImageNet ResNet-50 weights into the image so training never depends
# on a download at run time (and every run uses the identical weights file).
RUN python -c "import torchvision; torchvision.models.resnet50(weights='IMAGENET1K_V2')"

WORKDIR /workspace
