#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Note that usually LoRA needs to use larger learning rate
OUTPUT=$1
ZERO_STAGE=$2
HOME=/home/WangXu
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=0
fi
mkdir -p $OUTPUT
deepspeed --include localhost:3 main.py \
    --data_path Dahoas-rm-static-local \
    --model_name_or_path $HOME/huggingface_model/facebook/opt-1.3b \
    --gradient_accumulation_steps 4 \
    --lora_dim 128 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --max_seq_len 256 \
    --zero_stage $ZERO_STAGE \
    --deepspeed \
    --output_dir $OUTPUT &> $OUTPUT/training.log
