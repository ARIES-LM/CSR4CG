#!/bin/bash

#source ~/.bashrc
#conda activate dangle

RUN=/apdcephfs_cq2/share_47076/yongjingyin/Dangle

export PYTHONPATH=$RUN/fairseq
export PYTHONIOENCODING=utf8

MODEL=roberta

DATADIR=$RUN/cfq # path/to/cfq

SPLIT=${1}

DATA=${DATADIR}/mcd_data/cfq-${SPLIT}-fairseq

modelname=${2}

WORKDIR=$RUN/checkpoints/$modelname

#evalute
CUDA_VISIBLE_DEVICES=${3} python3 -u $RUN/myutils/eval_parsing.py $DATA \
--gen-subset test --path "$WORKDIR/checkpoint_best.pt" --dataset-impl raw \
--results-path $WORKDIR --quiet --max-sentences 200 >$WORKDIR/test.log 2>&1


