

#!/bin/bash


#SBATCH <SLURM OPTIONS> --nodes=128 --exclusive --ntasks-per-node=8 --job-name=megatron_gpt3_175b

export CUDA_DEVICE_MAX_CONNECTIONS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DIR=`pwd`

CHECKPOINT_PATH=/mnt/zero-bubble-pipeline-parallelism/Models/gpt-2/checkpoint
VOCAB_FILE=/mnt/zero-bubble-pipeline-parallelism/Models/gpt-2/data/gpt2-vocab.json
MERGE_FILE=/mnt/zero-bubble-pipeline-parallelism/Models/gpt-2/data/gpt2-merges.txt
DATA_PATH=/mnt/zero-bubble-pipeline-parallelism/Models/gpt-2/data/meg-gpt2_text_document

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --split 949,50,1
"

# Running locally
if [ -z "$WORLD_SIZE" ]; then
  export WORLD_SIZE=1
  export RANK=0
  export MASTER_ADDR=localhost
  export MASTER_PORT=10086
fi

if [ -z "$GPUS_PER_NODE" ]; then
  GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
fi

# if [ -z "$EXIT_INTERVAL" ]; then
#   EXIT_INTERVAL=1000
# fi

if [ -z "$LOG_INTERVAL" ]; then
  LOG_INTERVAL=10
fi

WORLD_SIZE_IN_GPUS=$(( $WORLD_SIZE * $GPUS_PER_NODE ))

if [ -z "$PIPELINE_SIZE" ]; then
  PIPELINE_SIZE=2
  LAYERS=24
  MICRO_BATCH_SIZE=4
  GLOBAL_BATCH_SIZE=16
  HIDDEN_SIZE=1024
  ATTENTION_HEADS=16
  # ZBH2
  ZERO_BUBBLE_MEM_LIMIT=$((2 * $PIPELINE_SIZE))
fi

# GPT_ARGS="
#     --num-layers 24 \
#     --hidden-size 1024 \
#     --num-attention-heads 16 \
#     --seq-length 1024 \
#     --max-position-embeddings 1024 \
#     --micro-batch-size 4 \
#     --global-batch-size 16 \
#     --lr 0.00015 \
#     --train-iters 500 \
#     --lr-decay-iters 320 \
#     --lr-decay-style cosine \
#     --min-lr 1.0e-5 \
#     --weight-decay 1e-2 \
#     --lr-warmup-fraction .01 \
#     --clip-grad 1.0 \
#     --bf16   
# "

# profile_ranks="0"
# for ((i = 1; i < $WORLD_SIZE_IN_GPUS; i++)); do
#     profile_ranks="$profile_ranks $i"
# done
# if [ -z "$ZERO_BUBBLE_TIMER_START" ]; then
#   ZERO_BUBBLE_TIMER_START=100
#   ZERO_BUBBLE_TIMER_END=110
# fi

if [ -z "$EVAL_INTERVAL" ]; then
  EVAL_INTERVAL=1000
fi

# OUTPUT_ARGS="
#     --log-interval 100 \
#     --save-interval 10000 \
#     --eval-interval 1000 \
#     --eval-iters 10
# "
# if [ -z "$TP_SIZE" ]; then
#   TP_SIZE=2
# fi


options=" \
  --pipeline-model-parallel-size $PIPELINE_SIZE \
  --num-layers $LAYERS \
  --hidden-size $HIDDEN_SIZE \
  --num-attention-heads $ATTENTION_HEADS \
  --seq-length 1024 \
  --max-position-embeddings 1024 \
  --micro-batch-size $MICRO_BATCH_SIZE \
  --global-batch-size $GLOBAL_BATCH_SIZE \
  --train-iters 500 \
  --lr-decay-iters 320 \
  --lr-warmup-fraction .01 \
  --lr 0.00015 \
  --min-lr 1.0e-5 \
  --lr-decay-style cosine \
  --log-interval ${LOG_INTERVAL} \
  --eval-iters 10 \
  --eval-interval $EVAL_INTERVAL \
  --clip-grad 1.0 \
  --weight-decay 1e-2 \
  --untie-embeddings-and-output-weights \
  --enable-zb-runtime \
  --no-create-attention-mask-in-dataloader \
  --transformer-impl local \
  --use-legacy-models \
  --distributed-backend nccl \
  --no-barrier-with-level-1-timing
  "

# --adam-beta1 0.9 \
# --adam-beta2 0.95 \
# --init-method-std 0.006 \
# --use-legacy-models \
# --transformer-impl local \
# --no-barrier-with-level-1-timing \
# --use-distributed-optimizer \

if [ -z "$FP32" ]; then
  options="$options --fp16"
fi

if [ ! -z "$PROFILED" ]; then
  options="$options --profile"
fi

if [ ! -z "$ZERO_BUBBLE_V_SCHEDULE" ]; then
  options="$options --zero-bubble-v-schedule "
fi

if [ -z "$ENABLE_ZERO_BUBBLE" ]; then
  options="$options --enable-zero-bubble \
  --zero-bubble-max-pending-backward $ZERO_BUBBLE_MEM_LIMIT"
  # if [ -z "$FP32" ]; then
  #   if [ -z "$SYNC_OPTIMIZER" ]; then
  #       options="$options --enable-optimizer-post-validation"
  #   fi
  # fi
fi

if [ ! -z "$ENABLE_EXACTLY_NUMERIC_MATCH" ]; then
  options="$options --enable-exactly-numeric-match \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0"
fi

if [ ! -z "$INTERLEAVED_1F1B" ]; then
  options="$options --num-layers-per-virtual-pipeline-stage 1"
fi

run_cmd="torchrun --nnodes $WORLD_SIZE \
  --node_rank $RANK \
  --master_addr $MASTER_ADDR \
  --master_port $MASTER_PORT \
  --nproc_per_node=$GPUS_PER_NODE ${DIR}/pretrain_gpt.py $@ ${options} ${DATA_ARGS}"

if [ ! -z "$PROFILED" ]; then
  run_cmd="nsys profile -s none -t nvtx,cuda \
    --output $AIP_RUN_NAME.$RANK.nsys-rep \
    --force-overwrite true \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    $run_cmd"
fi

echo $run_cmd
# sleep 100000
eval $run_cmd
#eval $run_cmd > >(tee log.$AIP_RUN_NAME) 2>&1

set +x