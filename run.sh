#!/bin/bash

ACCELERATE=true
ENV="/scratch/c102400/conda_envs/bfly"
MODELS=(resnet50 resnet101 resnet152 \
        wide_resnet50_2 wide_resnet101_2 \
        resnext101_32x8d resnext50_32x4d resnext101_64x4d \
        regnet_y_16gf regnet_y_32gf regnet_y_128gf regnet_y_3_2gf \
        regnet_x_8gf regnet_x_16gf regnet_x_32gf regnet_x_3_2gf \
        densenet121 densenet169 densenet201 densenet161 \
        vgg19 vgg19_bn vgg16_bn \
        efficientnet_v2_s efficientnet_v2_m efficientnet_v2_l \
        vit_b_16 vit_b_32 vit_l_32 vit_l_16 vit_h_14 \
        swin_v2_t swin_v2_s swin_v2_b \
        maxvit_t)
#MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
MODEL="resnet152"
BATCH_SIZE=16  # Batch size *per device*
EPOCHS=50
BASE_LR=0.0004 # Scale about linearly with workers
# Not pure bash but: $(echo "$BASE_LR*$SLURM_NNODES*$SLURM_GPUS_PER_TASK" | bc)
TARGET_ACCURACY=0.98  # Early stopping not yet implemented
PATIENCE=2


if [[ $(hostname) == n0* ]]; then
    DATA_DIR="/scratch/c7701273"
    RESULTS_DIR=$SCRATCH
    module load cuda/12.2.1-gcc-13.2.0-m4ekvjj  # To compile CUDA ops
    module load Anaconda3/2023.10/miniconda-base-2023.10
    eval "$($UIBK_CONDA_DIR/bin/conda shell.bash hook)"
    conda activate $ENV
elif [[ $(hostname) == lrdn* ]]; then
    DATA_DIR="/leonardo_work/EUHPC_D12_020/alindner/datasets"
    RESULTS_DIR="/leonardo_work/EUHPC_D12_020/fbarkman/results/"
    module load profile/deeplrn
    module load cineca-ai/4.3.0
fi

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

# PyTorch DDP launched with torchrun -> NOT IMPLEMENTED
else
    torchrun \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --node_rank=$SLURM_PROCID \
        --nproc_per_node=$SLURM_GPUS_PER_TASK \
        --rdzv_id=$SLURM_JOB_ID \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        --rdzv_backend=static `# c10d does not work on LEO5`\
        run_ddp.py \
        --model $MODEL \
        --batch-size $BATCH_SIZE \
        --epochs $EPOCHS \
        --base-lr $BASE_LR \
        --target-accuracy $TARGET_ACCURACY \
        --patience $PATIENCE \
        --data-dir $DATA_DIR \
        --results-dir $RESULTS_DIR
fi
