#!/bin/bash
RUN=/apdcephfs_cq2/share_47076/yongjingyin/fairseq-0.10multi
export PYTHONPATH=$RUN
export PYTHONIOENCODING=utf8

split=cogs

use_ckp=checkpoint_best.pt

#use_ckp=checkpoint_last.pt

datapath=/apdcephfs_cq2/share_47076/yongjingyin/procgdata/cogsv2/prep_data/gen_splits/gendevbin

#if [ $split = rand ];then
#    use_ckp=checkpoint_best.pt
#else
#    use_ckp=checkpoint_last.pt
#fi

modelname=${1}
GEN=$RUN/checkpoints/$modelname/$split.out

CUDA_VISIBLE_DEVICES=${2} python -u $RUN/fairseq_cli/generate.py $datapath \
    --gen-subset test \
    --path $RUN/checkpoints/$modelname/$use_ckp \
    --batch-size 512 --beam 1 >$GEN

SYS=$GEN.sys
REF=$GEN.ref

grep ^H $GEN | cut -f3-  > $SYS
grep ^T $GEN | cut -f2-  > $REF

python3 $RUN/comp_acc.py $SYS $REF > $RUN/checkpoints/$modelname/$split.acc

#python comp_bleu.py $SYS $REF > checkpoints/$modelname/random_test.result
