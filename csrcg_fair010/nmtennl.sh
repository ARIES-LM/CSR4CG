#!/bin/bash

RUN=/apdcephfs_cq2/share_47076/yongjingyin/fairseq-0.10multi
export PYTHONPATH=$RUN
export PYTHONIOENCODING=utf8
export OMP_NUM_THREADS=1

scale=tiny
# or small

DATA="/apdcephfs_cq2/share_47076/yongjingyin/entudata/${scale}v2bin"


if [ $scale == tiny ]; then
    valid_step=1000
    total_step=200000
    gpu="0,1,2,3"
    bz=4096
    NUM_GPU=4
elif [ $scale == small ]; then
    valid_step=2000
    total_step=500000
    gpu="0,1,2,3"
    bz=8192
    NUM_GPU=4
fi

# 0.07 or 0.1
tau=0.1
jslamb=3.0

modelname=ennl${scale}dp${dp}cregtokenjs${jslamb}cl${side}${classvar}tau${tau}warm${clwarm}

MODEL="$RUN/checkpoints/${modelname}"

echo "Start training..."
mkdir -p $MODEL

CUDA_VISIBLE_DEVICES=$gpu python -m torch.distributed.launch --master_port $RANDOM --nproc_per_node ${NUM_GPU} $RUN/fairseq_cli/train.py $DATA \
        --max-update ${total_step} \
        --max-source-positions 128 \
        --seed 1 \
        --clnspecial 5 \
        --var js --jslamb 3.0 --augnum4ce 2 --augnum 2 \
        --clgather 0 --classvar 1.0 --classvar_side tgt --intvcl 0.1 \
        --clwarm 4000 --cldim 128 \
        --patience 10 --no-epoch-checkpoints \
        --task translation --arch transformer_intv_wmt_en_de \
        --criterion label_smoothed_cross_entropy_set --label-smoothing 0.1 \
        --share-all-embeddings \
        --dropout 0.3 \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
        --weight-decay 0.0001 \
        --max-tokens $bz --save-dir $MODEL \
        --update-freq 1 --no-progress-bar --log-interval 200 \
        --keep-best-checkpoints 1 \
        --ddp-backend=legacy_ddp \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.0, "max_len_b": 50, "lenpen":0.6}' \
        --eval-bleu-detok moses --eval-bleu-remove-bpe \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
        --fp16 >>${MODEL}/train.log 2>&1

# test
test_model=checkpoint_best.pt
CUDA_VISIBLE_DEVICES=0 python $RUN/fairseq_cli/generate.py $DATA \
        --path $MODEL/$test_model \
        --max-len-a 1 --max-len-b 50 \
        --batch-size 128 --beam 5 --remove-bpe --lenpen 0.6 >>${MODEL}/test-$test_model.log 2>&1

# todo sacrebleu

