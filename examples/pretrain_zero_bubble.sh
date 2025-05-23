#!/bin/bash


#SBATCH <SLURM OPTIONS> --nodes=128 --exclusive --ntasks-per-node=8 --job-name=megatron_gpt3_175b

export CUDA_DEVICE_MAX_CONNECTIONS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`
mkdir -p $DIR/logs

DATASET_DIR='/tmp/zb_sample_dataset'
DATASET="${DATASET_DIR}/dataset/c4_text_document"
TOKENIZER="${DATASET_DIR}/tokenizers/tokenizer.model"

if [ ! -e "$DATASET"".idx" ]; then
  wget https://hf-mirror.com/datasets/ufotalent/zero_bubble_sample_dataset/resolve/main/zb_sample_dataset.tar.gz
  tar -xvf zb_sample_dataset.tar.gz -C /tmp
fi

# Running locally
if [ -z "$WORLD_SIZE" ]; then
  export WORLD_SIZE=1
  export RANK=0
  export MASTER_ADDR=localhost
  export MASTER_PORT=10086
fi

if [ -z "$GPUS_PER_NODE" ]; then
  GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
  # GPUS_PER_NODE=1
fi

if [ -z "$EXIT_INTERVAL" ]; then
  EXIT_INTERVAL=1000
fi

if [ -z "$LOG_INTERVAL" ]; then
  LOG_INTERVAL=10
fi

WORLD_SIZE_IN_GPUS=$(( $WORLD_SIZE * $GPUS_PER_NODE ))

if [ -z "$PIPELINE_SIZE" ]; then
  PIPELINE_SIZE=2
  LAYERS=8
  MICRO_BATCH_SIZE=4
  GLOBAL_BATCH_SIZE=16
  HIDDEN_SIZE=1024
  ATTENTION_HEADS=16
  ZERO_BUBBLE_MEM_LIMIT=$((2 * $PIPELINE_SIZE))
fi


if [ -z "$EVAL_INTERVAL" ]; then
  EVAL_INTERVAL=10000
fi

if [ -z "$TP_SIZE" ]; then
  TP_SIZE=2
fi

GQA=1

options=" \
  --pipeline-model-parallel-size $PIPELINE_SIZE \
  --num-layers $LAYERS \
  --hidden-size $HIDDEN_SIZE \
  --num-attention-heads $ATTENTION_HEADS \
  --exit-interval $EXIT_INTERVAL \
  --seq-length 1024 \
  --max-position-embeddings 1024 \
  --micro-batch-size $MICRO_BATCH_SIZE \
  --global-batch-size $GLOBAL_BATCH_SIZE \
  --group-query-attention \
  --num-query-groups $GQA \
  --train-iters 500 \
  --lr-decay-iters 320 \
  --lr-warmup-fraction .01 \
  --lr 6.0e-5 \
  --min-lr 6.0e-6 \
  --lr-decay-style cosine \
  --log-interval ${LOG_INTERVAL} \
  --eval-iters 40 \
  --eval-interval $EVAL_INTERVAL \
  --data-path ${DATASET} \
  --tokenizer-type GPTSentencePieceTokenizer \
  --tokenizer-model ${TOKENIZER} \
  --split 98,2,0 \
  --clip-grad 8.0 \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --init-method-std 0.006 \
  --no-barrier-with-level-1-timing \
  --untie-embeddings-and-output-weights \
  --use-legacy-models \
  --transformer-impl local \
  --enable-zb-runtime \
  --no-create-attention-mask-in-dataloader \
  "

if [ -z "$FP32" ]; then
  options="$options --fp16"
fi

if [ ! -z "$PROFILED" ]; then
  options="$options --profile"
fi

if [ ! -z "$ZERO_BUBBLE_V_SCHEDULE" ]; then
  ENABLE_ZERO_BUBBLE=1
  options="$options --zero-bubble-v-schedule "
fi

if [  -z "$ENABLE_ZERO_BUBBLE" ]; then
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
  --nproc_per_node=$GPUS_PER_NODE ${DIR}/pretrain_gpt.py $@ ${options} ${EXTRA_OPTIONS}"

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
