#!/bin/bash
RUN=./

export PYTHONPATH=$RUN
export PYTHONIOENCODING=utf8

split=${1}

datapath=procgdata/cfq/cfq-${split}-fairseq-pro-bin/

modelname=${2}

GEN=$RUN/checkpoints/$modelname/$split.out

CUDA_VISIBLE_DEVICES=${3} python -u $RUN/fairseq_cli/generate.py $datapath \
    -s word -t processed.predicate \
    --gen-subset test \
    --path $RUN/checkpoints/$modelname/checkpoint_best.pt \
    --batch-size 512 --beam 1 >$GEN

SYS=$GEN.sys
REF=$GEN.ref

grep ^H $GEN | cut -f3-  > $SYS
grep ^T $GEN | cut -f2-  > $REF

python3 $RUN/comp_acc.py $SYS $REF > $RUN/checkpoints/$modelname/$split.acc

