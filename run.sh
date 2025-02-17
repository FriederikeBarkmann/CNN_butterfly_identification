#!/bin/bash

ACCELERATE=true
ENV="conda_envs/bfly" #environment 
MODEL="resnet152"
BATCH_SIZE=16  # Batch size *per device*
EPOCHS=50
BASE_LR=0.0004 # Scale about linearly with workers
# Not pure bash but: $(echo "$BASE_LR*$SLURM_NNODES*$SLURM_GPUS_PER_TASK" | bc)
PATIENCE=2


DATA_DIR= <path> # path to dataset
RESULTS_DIR= <path> # path for results
module load profile/deeplrn
module load cineca-ai/4.3.0

export MASTER_PORT=24998
export MASTER_ADDR=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)

#export ACCELERATE_DEBUG_MODE="1"

# Keep track of settings
if [[ $(hostname) == "$MASTER_ADDR"* ]]; then
    echo
    echo "Using Accelerate: $ACCELERATE"
    echo "Model: $MODEL"
    echo "Batch size: $BATCH_SIZE"
    echo "Base learning rate: $BASE_LR"
    echo
fi

# DPP with HF Accelerate launched with accelerate launch
if [ "$ACCELERATE" = true ] ; then
    accelerate launch \
        --multi_gpu \
        --same_network \
        --machine_rank=$SLURM_PROCID \
        --main_process_ip=$MASTER_ADDR \
        --main_process_port=$MASTER_PORT \
        --num_machines=$SLURM_JOB_NUM_NODES \
        --num_processes=$(($SLURM_NNODES*$SLURM_GPUS_PER_TASK)) \
        --num_cpu_threads_per_process=$(($SLURM_CPUS_PER_TASK/$SLURM_GPUS_PER_TASK)) \
        --rdzv_backend=static `# c10d does not work on LEO5`\
        --mixed_precision="no" \
        --dynamo_backend="no" \
        run_acc.py \
        --model $MODEL \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --base-lr $BASE_LR \
        --target-accuracy $TARGET_ACCURACY \
        --patience $PATIENCE \
        --data-dir $DATA_DIR \
        --results-dir $RESULTS_DIR
fi
