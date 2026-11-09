#!/bin/bash -l
#SBATCH --job-name=ms1_frm
#SBATCH --partition=gpu-2h
#SBATCH --gpus=1
#SBATCH --ntasks-per-node=4
#SBATCH --output=%x_%j.log

# Usage, from the home folder that contains MS1/ and pml.sif:
#   sbatch --partition=gpu-test MS1/run_train.sbatch smoke
#   sbatch MS1/run_train.sbatch baselines | noaug | aug   (one job each; they can run in parallel)
#   sbatch --partition=gpu-5h MS1/run_train.sbatch long    (bigger model, 6 epochs)
MODE=${1:-noaug}
cd "$HOME"
container=${CONTAINER:-$HOME/pml.sif}
run="apptainer run --nv --bind /home/space/datasets $container python -m"
train="$run MS1.train --n-train 100000 --n-test 2000 --epochs 3 --lr 2e-3 --warmup 200 --eval-every 250"

case "$MODE" in
    smoke)
        $run MS1.baselines --n-fit 2000 --n-test 200 --n-solver 50 --out MS1/results/smoke/baselines.json
        $run MS1.train --n-train 64000 --n-test 500 --n-probe 500 --epochs 1 \
            --eval-every 100 --warmup 50 --lr 2e-3 --out-dir MS1/results/smoke/train
        ;;
    baselines)
        $run MS1.baselines --n-test 1000 --n-solver 300 --out MS1/results/baselines.json
        ;;
    noaug)
        $train --out-dir MS1/results/run_noaug
        ;;
    aug)
        $train --augment --out-dir MS1/results/run_aug
        ;;
    long)
        $run MS1.train --n-train 100000 --n-test 2000 --epochs 6 --lr 2e-3 --warmup 300 --eval-every 500 \
            --d-model 192 --n-layers 2 --n-steps 12 --out-dir MS1/results/run_long
        ;;
    *)
        echo "unknown mode: $MODE" >&2
        exit 1
        ;;
esac
