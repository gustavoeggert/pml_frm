#!/bin/bash -l
#SBATCH --job-name=ms1_frm
#SBATCH --partition=gpu-2h
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=%x_%j.log

# Usage, from the home folder that contains MS1/ and pml.sif:
#   sbatch --partition=gpu-test MS1/run_train.sbatch smoke
#   sbatch MS1/run_train.sbatch full
MODE=${1:-full}
cd "$HOME"
container=${CONTAINER:-$HOME/pml.sif}
run="apptainer run --nv --bind /home/space/datasets $container python -m"

if [ "$MODE" = "smoke" ]; then
    $run MS1.baselines --n-fit 2000 --n-test 200 --n-solver 50 --out MS1/results/smoke/baselines.json
    $run MS1.train --n-train 64000 --n-test 500 --n-probe 500 --epochs 1 \
        --eval-every 100 --warmup 50 --lr 2e-3 --out-dir MS1/results/smoke/train
else
    $run MS1.baselines --n-test 1000 --n-solver 300 --out MS1/results/baselines.json
    $run MS1.train --n-train 100000 --n-test 2000 --epochs 3 --augment --out-dir MS1/results/run_aug
    $run MS1.train --n-train 100000 --n-test 2000 --epochs 3 --out-dir MS1/results/run_noaug
fi
