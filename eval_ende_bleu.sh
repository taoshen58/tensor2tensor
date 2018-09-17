#!/usr/bin/env bash
DATA_DIR=/home/tshen/t2t/nmt/data
RUN_DIR=/home/tshen/t2t/nmt/$1

# For average checkpoint
CKPT_DIR=$RUN_DIR
if [ $# -gt 1 ]; then
    if [ $2 -gt 1 ]; then
        CKPT_DIR=$RUN_DIR/averaged_ckpt
        t2t-avg-all --model_dir $RUN_DIR -output_dir $CKPT_DIR --n $2
    fi
fi

t2t-decoder \
  --data_dir=~/t2t/nmt/data \
  --problem=translate_ende_wmt32k \
  --model=transformer \
  --hparams_set=transformer_base_single_gpu \
  --output_dir=${CKPT_DIR} \
  --decode_hparams=beam_size=4,alpha=0.6,batch_size=160, \
  --decode_from_file=${DATA_DIR}/newstest2014.en \
  --decode_reference=${DATA_DIR}/newstest2014.de \
  --decode_to_file=${RUN_DIR}/translation.en

t2t-bleu --translation=${RUN_DIR}/translation.en --reference=${DATA_DIR}/newstest2014.de