#!/bin/bash

RUN=/apdcephfs_cq2/share_47076/yongjingyin/fairseq-0.10multi
export PYTHONPATH=$RUN
export PYTHONIOENCODING=utf8
export OMP_NUM_THREADS=1

datapath=/apdcephfs_cq2/share_47076/yongjingyin/procgdata/CoGnition/data/bin

MODEL=$RUN/checkpoints

#  fairseq_task.py      disable_iterator_cache = True

gpu=${1}
export CUDA_VISIBLE_DEVICES=${gpu}

numgpu=1

if [ $numgpu == 2 ];then
PYRUN="python3 -m torch.distributed.launch --nproc_per_node $numgpu --master_port $RANDOM"
else
PYRUN="python3 -u"
fi

side=tgt
classvar=0.1
tau=0.05
jslamb=3.0
augnum=2
#var=none
var=js
group=${2}
negnum=128
seed=1

for bz in 4096 6000
do

modelname=cgmtsd${seed}gsf${group}tau${tau}bz${bz}

#--restore-file $MODEL/pretrainckpl6/checkpoint8.pt --reset-optimizer \
#--ddp-backend=legacy_ddp \

mkdir -p $MODEL/$modelname

${PYRUN} $RUN/fairseq_cli/train.py \
    $datapath \
    --save-dir $MODEL/$modelname \
    --max-tokens $bz --seed $seed \
    --encoder-layers 4 --decoder-layers 4 \
    --group-shuffle $group --token-sentidx-file cgmt.tgt.token2idx.pkl \
    --task translation --criterion label_smoothed_cross_entropy_set \
    --var $var --jslamb $jslamb --augnum4ce $augnum4ce --augnum $augnum \
    --cldim 128 --clwarm 4000 --classvar $classvar --classvar_side $side --intvcl $tau \
    --varnegnum $negnum \
    --log-interval 50 --patience 10 \
    --arch transformer_intv_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 1e-4 \
    --label-smoothing 0.1 --max-update 100000 --no-epoch-checkpoints \
    >>$MODEL/$modelname/train.log 2>&1

bash $RUN/decodecgmore.sh $modelname $gpu

done

