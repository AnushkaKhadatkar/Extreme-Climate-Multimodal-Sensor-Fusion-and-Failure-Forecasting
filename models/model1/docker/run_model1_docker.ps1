<#
Set up and run Model 1 in Docker on a Windows GPU box (Docker Desktop + WSL2 +
NVIDIA GPU). Mirrors baseline/scripts/run_baseline_docker.ps1.

Steps (each skipped if already done, unless forced):
  1. build   stf-openpcdet:pinned (from baseline/docker/, if missing) and
             stf-model1:pinned (from models/model1/docker/)
  2. data    Docker volume 'stf_kitti_mm' = the baseline's converted lidar,
             calib, labels and ImageSets + Model 1's real image_2
             (data/model1/image_2). A named volume, not a bind mount, so training
             reads from local disk (context.md section 8).
  3. infos   baseline/scripts/03_create_infos.py on that volume, with Model 1's
             dataset config
  4. run     a command inside the container (default: the unit tests)

Checkpoints and logs land in models/model1/output/ on the host (gitignored).

Usage (from the repo root):
  powershell -File models/model1/docker/run_model1_docker.ps1                      # build, data, infos, tests
  powershell -File models/model1/docker/run_model1_docker.ps1 -Cmd "bash /workspace/models/model1/tools/train_and_eval.sh"
  powershell -File models/model1/docker/run_model1_docker.ps1 -RefreshData         # recopy data + regenerate infos
#>
param(
    [string]$Cmd = "cd /workspace/models/model1 && python -m pytest -q tests",
    [switch]$RefreshData,
    [switch]$Detach,
    [string]$Name = "stf-model1",
    [string]$BaseImage = "stf-openpcdet:pinned",
    [string]$Image = "stf-model1:pinned",
    [string]$Volume = "stf_kitti_mm"
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$kitti = "$repo\baseline\data\kitti_format"
$images = "$repo\data\model1\image_2"
$output = "$repo\models\model1\output"
New-Item -ItemType Directory -Force $output | Out-Null

function Invoke-Docker { & docker @args; if ($LASTEXITCODE -ne 0) { throw "docker $($args -join ' ') failed ($LASTEXITCODE)" } }

# 1. images
docker image inspect $BaseImage *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "=== building $BaseImage from baseline/docker (first time: ~20-30 min) ==="
    Invoke-Docker build -t $BaseImage "$repo\baseline\docker"
}
docker image inspect $Image *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "=== building $Image ==="
    Invoke-Docker build -t $Image "$repo\models\model1\docker"
}

# 2. data volume
docker volume inspect $Volume *> $null
$fresh = $LASTEXITCODE -ne 0
if ($fresh -or $RefreshData) {
    if (-not (Test-Path "$kitti\training\velodyne")) { throw "$kitti not found -- run the baseline conversion (CLAUDE.md) first" }
    if (-not (Test-Path $images)) { throw "$images not found -- run models/model1/data_prep/01-03 first" }
    Write-Host "=== copying lidar/calib/labels + real images into volume $Volume (~45 GB) ==="
    if ($RefreshData -and -not $fresh) { Invoke-Docker volume rm $Volume }
    Invoke-Docker volume create $Volume | Out-Null
    # The baseline's image_2 holds placeholders: copy everything else, then the real images.
    $copy = "mkdir -p /dst/training && cp -a /src/ImageSets /dst/ && " +
            "for d in velodyne calib label_2; do cp -a /src/training/`$d /dst/training/; done && " +
            "cp -a /img /dst/training/image_2 && ls /dst/training && ls /dst/training/image_2 | wc -l"
    Invoke-Docker run --rm -v "${kitti}:/src:ro" -v "${images}:/img:ro" -v "${Volume}:/dst" $Image bash -c $copy
}

$common = @(
    "--gpus", "all", "--shm-size", "16g",
    "-v", "${Volume}:/opt/OpenPCDet/data/kitti_format_mm",
    "-v", "${output}:/opt/OpenPCDet/output",
    "-v", "${repo}:/workspace:ro",
    "-e", "PYTHONPATH=/workspace/models/model1"
)
# OpenPCDet resolves _BASE_CONFIG_ relative to tools/, so configs are deployed there.
$deployCfgs = "mkdir -p /opt/OpenPCDet/tools/cfgs/model1 && " +
              "cp /workspace/models/model1/configs/stf_multimodal_dataset.yaml /opt/OpenPCDet/tools/cfgs/dataset_configs/ && " +
              "(cp /workspace/models/model1/configs/model1_*.yaml /opt/OpenPCDet/tools/cfgs/model1/ 2>/dev/null || true)"

# 3. infos (skipped if all five already exist in the volume)
$haveInfos = docker run --rm -v "${Volume}:/d" $Image bash -c "ls /d/kitti_infos_{train,val,test_clear,test_light_fog,test_dense_fog}.pkl >/dev/null 2>&1 && echo yes"
if ($haveInfos -ne "yes") {
    Write-Host "=== generating infos for $Volume ==="
    Invoke-Docker run --rm @common $Image bash -c "$deployCfgs && python /workspace/baseline/scripts/03_create_infos.py --cfg /workspace/models/model1/configs/stf_multimodal_dataset.yaml --data-path /opt/OpenPCDet/data/kitti_format_mm --workers 8"
}

# 4. run
docker rm -f $Name *> $null
if ($Detach) {
    Invoke-Docker run -d --name $Name @common $Image bash -c "$deployCfgs && $Cmd" | Out-Null
    Write-Host "=== started container '$Name' -- follow with: docker logs -f $Name ==="
} else {
    Invoke-Docker run --rm --name $Name @common $Image bash -c "$deployCfgs && $Cmd"
}
