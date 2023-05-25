#!/bin/bash

RUN=/apdcephfs_cq2/share_47076/yongjingyin/fairseq-0.10multi
export PYTHONPATH=$RUN
export PYTHONIOENCODING=utf8
export OMP_NUM_THREADS=1

MODEL=$RUN/checkpoints

#mode='base'

mode='csr'

# inddev
#datapath=/apdcephfs_cq2/share_47076/yongjingyin/procgdata/cogsv2/prep_data/gen_splits/devbin

# ooddev
datapath=/apdcephfs_cq2/share_47076/yongjingyin/procgdata/cogsv2/prep_data/gen_splits/gendevbin

if [ $mode == 'base' ];then

seed=1

modelname=cogssd${seed}

mkdir -p $MODEL/$modelname
gpu=1
CUDA_VISIBLE_DEVICES=$gpu python3 -u fairseq_cli/train.py \
    $datapath \
    --universal 0 \
    --encoder-layers 2 --decoder-layers 2 \
    --encoder-embed-dim 512 --encoder-ffn-embed-dim 512 \
    --attention-dropout 0.1 --activation-dropout 0.0 --dropout 0.1 \
    --decoder-embed-dim 512 --decoder-ffn-embed-dim 512 \
    --encoder-attention-heads 8 --decoder-attention-heads 8 \
    --seed $seed \
    --lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --save-dir $MODEL/$modelname \
    --task cg --criterion cross_entropy \
    --log-interval 50 \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --clip-norm 1.0 \
    --weight-decay 0.0 --max-tokens 4096 --max-update 60000 --no-epoch-checkpoints \
    --max-sentences-valid 512 \
    --validate-interval 5 --eval-bleu --eval-bleu-args '{"beam":1}' --best-checkpoint-metric acc --maximize-best-checkpoint-metric \
    >>$MODEL/$modelname/train.log 2>&1

    bash decodecog.sh $modelname $gpu

elif [ $mode == 'csr' ];then

var=js

gpu=${1}

seed=42

bz=4096
bz=6000
classvar=0.01

for gsf in ${2}
do

modelname=cogs_bz${bz}gsf${gsf}sd${seed}

mkdir -p $MODEL/$modelname

#--disable-validation \

CUDA_VISIBLE_DEVICES=$gpu python3 -u $RUN/fairseq_cli/train.py \
    $datapath \
    --group-shuffle $gsf --token-sentidx-file "cogs.token_idx.pkl" \
    --task cg --criterion cross_entropyv2 \
    --var $var --jslamb 1.0 --augnum4ce 2 --augnum 2 \
    --clnspecial 2 --classvar $classvar --classvar_side tgt --intvcl 0.07 \
    --universal 0 \
    --validate-after-updates 30000 \
    --encoder-layers 2 --decoder-layers 2 \
    --encoder-embed-dim 512 --encoder-ffn-embed-dim 512 \
    --attention-dropout 0.1 --activation-dropout 0.0 --dropout 0.1 \
    --decoder-embed-dim 512 --decoder-ffn-embed-dim 512 \
    --encoder-attention-heads 8 --decoder-attention-heads 8 \
    --seed $seed \
    --lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --save-dir $MODEL/$modelname \
    --log-interval 50 \
    --eval-bleu --eval-bleu-args '{"beam":1}' --best-checkpoint-metric acc --maximize-best-checkpoint-metric \
    --arch transformer_intv_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --clip-norm 1.0 \
    --weight-decay 0.0 --max-tokens 4096 --max-update 60000 --no-epoch-checkpoints \
    --max-sentences-valid 512 \
    --validate-interval 5 \
    >>$MODEL/$modelname/train.log 2>&1

bash $RUN/decodecog.sh $modelname $gpu

done

fi

