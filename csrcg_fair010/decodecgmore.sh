#!/bin/bash
export PYTHONIOENCODING=utf8

RUN=/apdcephfs_cq2/share_47076/yongjingyin/fairseq-0.10multi
export PYTHONPATH=$RUN

datapath=/apdcephfs/share_47076/yongjingyin/cogmt/data/cg-test/
evalpath=/apdcephfs/share_47076/yongjingyin/cogmt/eval

modelname=${1}

ckps=($(find $RUN/checkpoints/${modelname} -name "*best*.pt"))

for ckp in ${ckps[@]}
do
GEN=$ckp.cgtest.out
echo $GEN

CUDA_VISIBLE_DEVICES=${2} python3 -u $RUN/fairseq_cli/generate.py $datapath/bin \
    --path $ckp \
    --batch-size 256 --beam 5 --remove-bpe >$GEN

evalfile=$ckp.cgtest.src-hyp-tgt

python $RUN/_reorder.py $datapath/test.en $GEN $datapath/test.zh > $evalfile

cut -f3 $evalfile > $ckp.cgtest.ref
cut -f2 $evalfile > $ckp.cgtest.hyp

sacrebleu $ckp.cgtest.ref -i $ckp.cgtest.hyp -tok zh > $ckp.cgtest.bleu

paste ${datapath}/test.en ${datapath}/test.compound $ckp.cgtest.hyp >$ckp.cgtest.hyp.merge.txt

#bash spliterrorrate.sh $ckp.cgtest.hyp.merge.txt

python $evalpath/eval.py $ckp.cgtest.hyp.merge.txt $evalpath/lexicon >$ckp.cgacc.log

done



#SYS=$GEN.sys
#grep ^H $GEN | cut -f3-  > $SYS
#REF=$datapath/cg-test/test.zh
#python comp_bleu.py $SYS $REF > checkpoints/$modelname/cgtest.result
