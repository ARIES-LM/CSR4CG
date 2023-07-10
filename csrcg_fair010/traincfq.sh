#!/bin/bash

RUN=./
export PYTHONPATH=$RUN
export PYTHONIOENCODING=utf8
export OMP_NUM_THREADS=1

MODEL=$RUN/checkpoints

SPLIT=mcd1

#SPLIT=random_split
#  mcd1 | mcd2 | mcd3

datapath=procgdata/cfq/cfq-${SPLIT}-fairseq-pro-bin/

export CUDA_VISIBLE_DEVICES=${1}

seed=1
#seed=2
#seed=3

modelname=cfq${SPLIT}

mkdir -p $MODEL/$modelname

python3 -u $RUN/fairseq_cli/train.py \
    $datapath \
    -s word -t processed.predicate \
    --universal 1 --dropout 0.1 \
    --encoder-embed-dim 256 --encoder-ffn-embed-dim 512 \
    --decoder-embed-dim 256 --decoder-ffn-embed-dim 512 \
    --encoder-attention-heads 4 --decoder-attention-heads 4 \
    --seed $seed \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --save-dir $MODEL/$modelname \
    --clwarm 4000 --classvar 0.4 --classvar_side tgt --intvcl 0.05 --cldim 64 \
    --task cg --criterion cross_entropyv2 --var js --augnum4ce 1 --jslamb 3.0 --augnum 2 \
    --log-interval 100 \
    --arch transformer_intv_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --weight-decay 0.00001 \
    --batch-size 256 --max-update 100000 --no-epoch-checkpoints \
    --max-sentences-valid 512 --validate-after-updates 60000 \
    --validate-interval 5 --eval-bleu --eval-bleu-args '{"beam":1}' --best-checkpoint-metric acc --maximize-best-checkpoint-metric \
    >>$MODEL/$modelname/train.log 2>&1

bash $RUN/decodecfq.sh $SPLIT $modelname $gpu

