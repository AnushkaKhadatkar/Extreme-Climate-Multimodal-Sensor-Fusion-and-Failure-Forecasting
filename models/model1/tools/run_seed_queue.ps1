<#
Runs the extra-seed M5p / M4bp pair for seed 222 once the seed-111 pair (already launched separately) has finished,
so the two pairs don't contend for the GPU. Polls docker logs for both seed-111 containers to report "finished",
then launches 222 the same way seed 111 was launched. Meant to be started once and left running; it blocks until
the 222 pair is itself finished and exits 0, so the caller (or a scheduled check) can tell the whole queue is done
by this script's own exit code.

Why a script and not another scheduled check: the 30-minute check pattern used for earlier runs works for "is this
stuck, resume it", but chaining "when A and B finish, start C and D" through many short poll turns is exactly what a
long-lived background process is for instead.
#>
param(
    [string]$Repo = (Resolve-Path "$PSScriptRoot\..\..\..").Path,
    [string]$ScratchM3 = "C:\Users\anushka\AppData\Local\Temp\claude\4bf23e8277f42785\4c87b1a6-00d4-4a3e-ae63-67bca7af2408\scratchpad\m3",
    [int]$PollSeconds = 120
)
$ErrorActionPreference = "Stop"

function Wait-Finished($names) {
    while ($true) {
        $done = $true
        foreach ($n in $names) {
            $log = docker logs $n 2>&1
            if (-not ($log | Select-String 'finished.*exit=')) { $done = $false }
        }
        if ($done) { return }
        Start-Sleep $PollSeconds
    }
}

Write-Host "Waiting for seed-111 pair (m1-m5p-s111, m1-m4bp-s111) to finish..."
Wait-Finished @("m1-m5p-s111", "m1-m4bp-s111")
Write-Host "Seed-111 pair finished. Launching seed 222."

$out = "$Repo\models\model1\output"
foreach ($x in @(@("m1-m5p-s222", "m5p_s222.sh"), @("m1-m4bp-s222", "m4bp_s222.sh"))) {
    docker rm -f $x[0] *> $null
    docker run -d --name $x[0] --gpus all --shm-size 16g `
        -v stf_kitti_mm:/opt/OpenPCDet/data/kitti_format_mm `
        -v "${out}:/opt/OpenPCDet/output" `
        -v "${Repo}:/workspace:ro" `
        -v "${ScratchM3}:/m3:ro" `
        -e PYTHONPATH=/workspace/models/model1 `
        stf-model1:pinned bash "/m3/$($x[1])" | Out-Null
}
Write-Host "Seed-222 pair launched: $(Get-Date -Format s)"

Wait-Finished @("m1-m5p-s222", "m1-m4bp-s222")
Write-Host "Seed-222 pair finished: $(Get-Date -Format s)"
