<#
Run the baseline reproduction end to end in the pinned Docker image on a Windows
GPU box (Docker Desktop + WSL2 + NVIDIA GPU). See baseline/README.md.

Steps (each skipped if already done, unless forced):
  1. build    docker image stf-openpcdet:pinned from baseline/docker/
  2. data     copy baseline/data/kitti_format into Docker volume 'stf_kitti'
              (a named volume, not a bind mount: bind mounts from Windows go
              through WSL's file share and would bottleneck every epoch -- the
              context.md section 8 "train off local disk" rule)
  3. infos    03_create_infos.py inside the container (infos + gt database)
  4. train    04_train_and_eval.sh in a detached container named 'stf-baseline';
              follow with: docker logs -f stf-baseline

Checkpoints and logs land in baseline/output/ on the host (gitignored).

Usage (from the repo root):
  powershell -File baseline/scripts/run_baseline_docker.ps1               # full run
  powershell -File baseline/scripts/run_baseline_docker.ps1 -Smoke        # 1-epoch smoke test, tag 'smoke'
  powershell -File baseline/scripts/run_baseline_docker.ps1 -RefreshData  # recopy data + regenerate infos
#>
param(
    [switch]$Smoke,
    [switch]$RefreshData,
    [switch]$Foreground,
    [string]$Image = "stf-openpcdet:pinned",
    [string]$Volume = "stf_kitti"
)
$ErrorActionPreference = "Stop"
$repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$kitti = "$repo\baseline\data\kitti_format"
$output = "$repo\baseline\output"
New-Item -ItemType Directory -Force $output | Out-Null

function Invoke-Docker { & docker @args; if ($LASTEXITCODE -ne 0) { throw "docker $($args -join ' ') failed ($LASTEXITCODE)" } }

# 1. image
docker image inspect $Image *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host "=== building $Image (first time: ~20-30 min) ==="
    Invoke-Docker build -t $Image "$repo\baseline\docker"
}

# 2. data volume
docker volume inspect $Volume *> $null
$fresh = $LASTEXITCODE -ne 0
if ($fresh -or $RefreshData) {
    if (-not (Test-Path "$kitti\training\velodyne")) { throw "$kitti not found -- run 01/02 conversion first" }
    Write-Host "=== copying kitti_format into volume $Volume (~21 GB) ==="
    if ($RefreshData -and -not $fresh) { Invoke-Docker volume rm $Volume }
    Invoke-Docker volume create $Volume | Out-Null
    Invoke-Docker run --rm -v "${kitti}:/src:ro" -v "${Volume}:/dst" $Image bash -c "cp -a /src/. /dst/ && ls /dst/ImageSets"
}

$common = @(
    "--gpus", "all", "--shm-size", "16g",
    "-v", "${Volume}:/opt/OpenPCDet/data/kitti_format",
    "-v", "${output}:/opt/OpenPCDet/output",
    "-v", "${repo}:/workspace:ro"
)
$deployCfgs = "cp /workspace/baseline/configs/pv_rcnn_stf.yaml /opt/OpenPCDet/tools/cfgs/kitti_models/ && " +
              "cp /workspace/baseline/configs/kitti_dataset_stf.yaml /opt/OpenPCDet/tools/cfgs/dataset_configs/"

# 3. infos (skipped if all five already exist in the volume)
$haveInfos = docker run --rm -v "${Volume}:/d" $Image bash -c "ls /d/kitti_infos_{train,val,test_clear,test_light_fog,test_dense_fog}.pkl /d/kitti_dbinfos_train.pkl >/dev/null 2>&1 && echo yes"
# -contains, not -eq: the CUDA base image prints a banner to stdout first, so
# $haveInfos is an array of lines ending in "yes".
if (-not ($haveInfos -contains "yes")) {
    Write-Host "=== generating infos + gt database ==="
    Invoke-Docker run --rm @common $Image bash -c "$deployCfgs && python /workspace/baseline/scripts/03_create_infos.py --cfg /workspace/baseline/configs/kitti_dataset_stf.yaml --data-path /opt/OpenPCDet/data/kitti_format --workers 8"
}

# 4. train + select + evaluate
$envArgs = @("-e", "OPENPCDET_ROOT=/opt/OpenPCDet")
$name = "stf-baseline"
if ($Smoke) { $envArgs += @("-e", "EPOCHS=1", "-e", "EXTRA_TAG=smoke"); $name = "stf-baseline-smoke" }
docker rm -f $name *> $null
$cmd = "$deployCfgs && bash /workspace/baseline/scripts/04_train_and_eval.sh"
if ($Foreground) {
    Invoke-Docker run --rm --name $name @common @envArgs $Image bash -c $cmd
} else {
    Invoke-Docker run -d --name $name @common @envArgs $Image bash -c $cmd | Out-Null
    Write-Host "=== started container '$name' ==="
    Write-Host "follow:  docker logs -f $name"
    Write-Host "results: $output\kitti_models\pv_rcnn_stf\$(if ($Smoke) {'smoke'} else {'stf_baseline'})\baseline_logs\"
}
