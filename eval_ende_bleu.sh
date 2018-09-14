#!/usr/bin/env bash
DATA_DIR=/home/tshen/t2t/nmt/data
RUN_DIR=/home/tshen/t2t/nmt/$1

t2t-decoder \
  --data_dir=~/t2t/nmt/data \
  --problem=translate_ende_wmt32k \
  --model=transformer \
  --hparams_set=transformer_base_single_gpu \
  --output_dir=${RUN_DIR} \
  --decode_hparams=beam_size=4,alpha=0.6,batch_size=160, \
  --decode_from_file=${DATA_DIR}/newstest2014.en \
  --decode_reference=${DATA_DIR}/newstest2014.de \
  --decode_to_file=${RUN_DIR}/translation.en

t2t-bleu --translation=${RUN_DIR}/translation.en --reference=${DATA_DIR}/newstest2014.de