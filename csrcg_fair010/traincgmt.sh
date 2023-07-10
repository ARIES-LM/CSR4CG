#!/bin/bash

export PYTHONIOENCODING=utf8
export OMP_NUM_THREADS=1

datapath=CoGnition/data/bin

MODEL=$RUN/checkpoints

#  fairseq_task.py      disable_iterator_cache = True

gpu=${1}
export CUDA_VISIBLE_DEVICES=${gpu}

# 7 77
seed=1

modelname=cgmt

mkdir -p $MODEL/$modelname

${PYRUN} $RUN/fairseq_cli/train.py \
    $datapath \
    --save-dir $MODEL/$modelname \
    --max-tokens 4096 --seed $seed \
    --encoder-layers 4 --decoder-layers 4 \
    --task translation --criterion label_smoothed_cross_entropy_set \
    --var js --jslamb 3.0 --augnum4ce 1 --augnum 2 \
    --cldim 128 --clwarm 4000 --classvar 0.1 --classvar_side tgt --intvcl 0.05 --varnegnum 128 \
    --log-interval 50 --patience 10 \
    --arch transformer_intv_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 1e-4 \
    --label-smoothing 0.1 --max-update 100000 --no-epoch-checkpoints \
    >>$MODEL/$modelname/train.log 2>&1

bash $RUN/decodecgmore.sh $modelname $gpu

