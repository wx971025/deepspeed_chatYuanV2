#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
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

deepspeed --include localhost:2,3 main.py \
          --data_path Dahoas-rm-static-local\
          --model_name_or_path $HOME/huggingface_model/facebook/opt-350m \
          --num_padding_at_beginning 1 \
          --weight_decay 0.1 \
          --disable_dropout \
          --gradient_accumulation_steps 1 \
          --per_device_train_batch_size 32 \
          --per_device_eval_batch_size 32 \
          --max_seq_len 256 \
          --zero_stage $ZERO_STAGE \
          --deepspeed  \
          --output_dir $OUTPUT &> $OUTPUT/training.log
