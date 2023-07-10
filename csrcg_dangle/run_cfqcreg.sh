#!/bin/bash

RUN=./

export PYTHONPATH=$RUN/fairseq
export PYTHONIOENCODING=utf8

MODEL=roberta

DATADIR=$RUN/cfq # path/to/cfq

SPLIT=${1:-mcd1}

DATA=${DATADIR}/mcd_data/cfq-${SPLIT}-fairseq

#rm apply_bert_init

gpu=${2:-0}
SEED=42
#SEED=1
#SEED=3

modelname=${SPLIT}

WORKDIR=$RUN/checkpoints/$modelname
mkdir -p $WORKDIR

CUDA_VISIBLE_DEVICES=$gpu python3 -u $RUN/fairseq/fairseq_cli/train.py $DATA \
--roberta-path "$RUN/roberta.base" \
--fp16 \
--validate-after-epoch 50 \
--task semantic_parsing \
--arch transformer_creg_roberta \
--criterion cross_entropy_creg \
--var js --jslamb 1.0 --augnum4ce 1 --augnum 2 --validvar 0 \
--classvar 0.3 --classvar_side src --intvcl 0.07 \
--clwarm 4000 --cldim 128 \
--dataset-impl raw \
--optimizer adam --clip-norm 1.0 \
--lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
--max-tokens 8192 --max-update 50000 \
--save-dir $WORKDIR \
--weight-decay 0.001 \
--no-epoch-checkpoints \
--eval-accuracy \
--best-checkpoint-metric accuracy \
--maximize-best-checkpoint-metric \
--seed $SEED >>$WORKDIR/train.log 2>&1

#evalute
CUDA_VISIBLE_DEVICES=$gpu python3 -u $RUN/myutils/eval_parsing.py $DATA \
--gen-subset test --path "$WORKDIR/checkpoint_best.pt" --dataset-impl raw \
--results-path $WORKDIR --quiet --max-sentences 200 >$WORKDIR/test.log 2>&1

