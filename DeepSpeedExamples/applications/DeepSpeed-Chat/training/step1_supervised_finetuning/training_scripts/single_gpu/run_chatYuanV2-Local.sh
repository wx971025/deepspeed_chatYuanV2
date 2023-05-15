#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
CHATYUANV2=/home/Lyuqi/RHLF/SFT/oursmodels/epochs0
CHATYUANV2_LOCAL=/home/WangXu/huggingface_model/chatYuan-our
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi
mkdir -p $OUTPUT
# --data_path wangrui6/Zhihu-KOL Cohere/miracl-zh-queries-22-12 Hello-SimpleAI/HC3-Chinese mkqa-Chinese \
# The Chinese data we found mostly only contain one response without another
# "rejected" response. Thus we only test the step 1 finetuning and use
# a data_split of 10,0,0 (keep all data for step 1).
deepspeed --include localhost:2,3 main.py \
   --data_path belle2M \
   --data_split 10,0,0 \
   --model_name_or_path $CHATYUANV2_LOCAL \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --max_seq_len 512 \
   --learning_rate 9.65e-6 \
   --weight_decay 0.1 \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234\
   --zero_stage $ZERO_STAGE \
   --deepspeed \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
