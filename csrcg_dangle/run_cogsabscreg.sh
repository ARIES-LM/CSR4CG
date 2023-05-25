#!/bin/bash

#source ~/.bashrc
#conda activate dangle

RUN=/apdcephfs_cq2/share_47076/yongjingyin/Dangle
export PYTHONPATH=$RUN/fairseq
export PYTHONIOENCODING=utf8

# transformer_relative | transformer_absolute | transformer_dangle_relative | transformer_dangle_absolute | roberta_dangle

MODEL=transformer_absolute
#MODEL=roberta_dangle

RECURSION=$3 # 2 | 3 | 4 | 5
#DATADIR=$4 # path/to/cogs


#DATA=${DATADIR}/prep_data/cogs-fairseq-recursion${RECURSION}
#BPE_DATA=${DATADIR}/prep_data/cogs-fairseq-recursion${RECURSION}-bpe

DATA=$RUN/COGS-main/prep_data/gen_splits
BPE_DATA=$RUN/COGS-main/prep_data/gen_splits-bpe

#if [[ "$MODEL" == transformer* && ! -f "glove.840B.300d.txt" ]] ; then
#	wget https://nlp.stanford.edu/data/glove.840B.300d.zip
#	unzip glove.840B.300d.zip
#fi

gpu=${1:-0}
SEED=42

modelname=cogsglovesd${SEED}

WORKDIR=$RUN/checkpoints/$modelname
mkdir -p $WORKDIR

#	--arch transformer_abs_cogs \
#  --save-interval-updates 5000 \
#  --keep-interval-updates 5 \
#  --encoder-embed-path glove.840B.300d.txt --decoder-embed-path glove.840B.300d.txt \

# train
CUDA_VISIBLE_DEVICES=$gpu python3 -u $RUN/fairseq/fairseq_cli/train.py $DATA \
--task semantic_parsing \
--arch transformer_creg_glove \
--encoder-embed-path $RUN/glove4cogs.src.pt --decoder-embed-path $RUN/glove4cogs.tgt.pt \
--glove-scale 4 --no-scale-embedding \
--criterion cross_entropy_creg \
--clnspecial 2 \
--var js --jslamb 1.0 --augnum4ce 2 --augnum 2 --validvar 0 \
--classvar 0.01 --classvar_side tgt --intvcl 0.07 --cldim 128 \
--log-interval 20 \
--dataset-impl raw \
--share-decoder-input-output-embed \
--optimizer adam --clip-norm 1.0 \
--lr 1e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
--validate-interval 5 \
--max-tokens 4096 --max-update 50000 \
--save-dir $WORKDIR \
--no-epoch-checkpoints --eval-accuracy --best-checkpoint-metric accuracy \
--maximize-best-checkpoint-metric \
--seed $SEED >>$WORKDIR/train.log 2>&1

#--results-path $WORKDIR
#evaluate

ckp=$WORKDIR/checkpoint_best.pt

CUDA_VISIBLE_DEVICES=$gpu python3 -u $RUN/myutils/eval_parsing.py $DATA \
--gen-subset test --path $ckp --dataset-impl raw \
--quiet --max-sentences 200 >>$WORKDIR/test.log 2>&1
