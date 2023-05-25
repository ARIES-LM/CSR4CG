#!/bin/bash

RUN=/apdcephfs_cq2/share_47076/yongjingyin/fairseq-0.10multi
export PYTHONPATH=$RUN
export PYTHONIOENCODING=utf8
export OMP_NUM_THREADS=1

MODEL=$RUN/checkpoints

SPLIT=mcd1

#SPLIT=random_split
#  mcd1 | mcd2 | mcd3
#mode='base'
mode='csr'

#mode='rl'

#mode='spa'

if [ $mode == 'base' ];then
# seperate vocab
datapath=/apdcephfs_cq2/share_47076/yongjingyin/procgdata/cfq/cfq-${SPLIT}-fairseq-pro-bin/

seed=1
modelname=cfq${SPLIT}sd${seed}
mkdir -p $MODEL/$modelname
gpu=${1}
#    --save-interval-updates 1000 --keep-interval-updates 1 \ --keep-last-epochs 5
    #--finetune-from-model checkpoints/basesd1bz4096lr5e/checkpoint_best.pt \
    #--lr-scheduler fixed --lr 1e-4 \
#--validate-interval 10
    #--attention-dropout 0.1 \
#--activation-dropout 0.1 \

CUDA_VISIBLE_DEVICES=$gpu python3 -u $RUN/fairseq_cli/train.py \
    $datapath \
    -s word -t processed.predicate \
    --universal 1 \
    --encoder-embed-dim 256 --encoder-ffn-embed-dim 512 \
    --dropout 0.1 \
    --decoder-embed-dim 256 --decoder-ffn-embed-dim 512 \
    --encoder-attention-heads 4 --decoder-attention-heads 4 \
    --seed $seed \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --save-dir $MODEL/$modelname \
    --task cg --criterion cross_entropy \
    --log-interval 100 --validate-after-updates 0 \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --weight-decay 0.00001 --batch-size 256 --max-update 100000 --no-epoch-checkpoints \
    --max-sentences-valid 512 \
    --validate-interval 5 --eval-bleu --eval-bleu-args '{"beam":1}' --best-checkpoint-metric acc --maximize-best-checkpoint-metric \
    >>$MODEL/$modelname/train.log 2>&1

    bash $RUN/decodecfq.sh $SPLIT $modelname $gpu

elif [ $mode == 'spa' ];then

# seperate vocab
datapath=/apdcephfs_cq2/share_47076/yongjingyin/procgdata/cfq/cfq-${SPLIT}-fairseq-pro-bin/

seed=1
bz=256

#lr=1e-4
#ffndim=1024

cut=0

modelname=cfq${SPLIT}spa

mkdir -p $MODEL/$modelname
gpu=${1}

CUDA_VISIBLE_DEVICES=$gpu python3 -u $RUN/fairseq_cli/train.py \
    $datapath \
    -s word -t processed.predicate \
    --universal 1 \
    --encoder-embed-dim 256 --encoder-ffn-embed-dim 512 \
    --dropout 0.1 \
    --decoder-embed-dim 256 --decoder-ffn-embed-dim 512 \
    --encoder-attention-heads 4 --decoder-attention-heads 4 \
    --seed $seed \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --save-dir $MODEL/$modelname \
    --task cg --criterion cross_entropy \
    --log-interval 100 --validate-after-updates 0 \
    --arch transformer_cut_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --weight-decay 0.00001 --batch-size $bz --max-update 100000 --no-epoch-checkpoints \
    --max-sentences-valid 512 \
    --validate-interval 5 --eval-bleu --eval-bleu-args '{"beam":1}' --best-checkpoint-metric acc --maximize-best-checkpoint-metric \
    >>$MODEL/$modelname/train.log 2>&1

#    bash $RUN/decodecfq.sh $SPLIT $modelname $gpu

elif [ $mode == 'csr' ];then

# seperate vocab
datapath=/apdcephfs_cq2/share_47076/yongjingyin/procgdata/cfq/cfq-${SPLIT}-fairseq-pro-bin/

bz=400
augnum=2
var=rd
aug4ce=1

export CUDA_VISIBLE_DEVICES=${1}

#export CUDA_VISIBLE_DEVICES=0

#mport=$RANDOM
varside=tgt
classvar=0.4
jslamb=3.0

seed=1

gsf=${2}

modelname=cfq${SPLIT}sd${seed}gsf${gsf}${varside}${classvar}bz${bz}

mkdir -p $MODEL/$modelname

# --save-interval-updates 1000 --keep-interval-updates 1 
#--finetune-from-model checkpoints/basesd1bz4096lr5e/checkpoint_best.pt \
#--lr-scheduler fixed --lr 1e-4 \
#--restore-file $MODEL/cfq${SPLIT}rd${jslamb}src0/checkpoint_best.pt \
#python3 -m torch.distributed.launch --nproc_per_node 2 --master_port $mport $RUN/fairseq_cli/train.py \

#--valid-subset valid --ignore-unused-valid-subsets \
#--attention-dropout 0.1 --activation-dropout 0.1 \

python3 -u $RUN/fairseq_cli/train.py \
    $datapath \
    -s word -t processed.predicate \
    --group-shuffle $gsf --token-sentidx-file "cfq.token_idx.pkl" \
    --universal 1 \
    --dropout 0.1 \
    --encoder-embed-dim 256 --encoder-ffn-embed-dim 512 \
    --decoder-embed-dim 256 --decoder-ffn-embed-dim 512 \
    --encoder-attention-heads 4 --decoder-attention-heads 4 \
    --seed $seed \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --save-dir $MODEL/$modelname \
    --clwarm 4000 --classvar $classvar --classvar_side $varside --intvcl 0.05 --cldim 64 \
    --task cg --criterion cross_entropyv2 --var $var --augnum4ce $aug4ce --jslamb $jslamb --augnum $augnum \
    --log-interval 100 \
    --arch transformer_intv_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --weight-decay 0.00001 \
    --batch-size $bz --max-update 100000 --no-epoch-checkpoints \
    --max-sentences-valid 512 --validate-after-updates 60000 \
    --validate-interval 5 --eval-bleu --eval-bleu-args '{"beam":1}' --best-checkpoint-metric acc --maximize-best-checkpoint-metric \
    >>$MODEL/$modelname/train.log 2>&1

bash $RUN/decodecfq.sh $SPLIT $modelname $gpu

elif [ $mode == 'rl' ];then

datapath=/apdcephfs_cq2/share_47076/yongjingyin/procgdata/cfq/cfq-${SPLIT}-fairseq-pro-bin/

seed=1
bz=512

vqlamb=${5}
vqdim=${4}

encl=6
decl=6
packhead=4
packxselfatt=0
decoderattx=0
decodergate=0

packnum=${2}
lr=5e-4

for normonq in 0
do
for k in ${3}
do

modelname=imscfq${SPLIT}dpbz${bz}pack${packnum}vq${k}dim${vqdim}lamb${vqlamb}

mkdir -p $MODEL/$modelname

gpu=${1}

#    --save-interval-updates 1000 --keep-interval-updates 1 \ --keep-last-epochs 5
#--attention-dropout 0.1 --activation-dropout 0.1
#--finetune-from-model checkpoints/basesd1bz4096lr5e/checkpoint_best.pt \
#--lr-scheduler fixed --lr 1e-4 \
#    --restore-file checkpoints/priortagxymidg${g}k${k}dim${vqdim}ema${emadecay}pack${packnum}cos$vqcos/checkpoint_best.pt \
#    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer \
#--cross-self-attention --no-cross-attention \


CUDA_VISIBLE_DEVICES=$gpu python3 -u fairseq_cli/train.py \
    $datapath \
    --save-dir $MODEL/$modelname \
    -s word -t processed.predicate \
    --validate-interval 5 \
    --eval-bleu --eval-bleu-args '{"beam":1}' --best-checkpoint-metric acc --maximize-best-checkpoint-metric \
    --universal 1 \
    --encoder-embed-dim 256 --encoder-ffn-embed-dim 512 \
    --decoder-embed-dim 256 --decoder-ffn-embed-dim 512 \
    --encoder-attention-heads 4 --decoder-attention-heads 4 \
    --encoder-layers $encl --decoder-layers $decl \
    --seed $seed \
    --dropout 0.1 \
    --share-decoder-input-output-embed --arch transformer_cgrl_iwslt \
    --packdim 256 --packnum $packnum --pack-iters $encl --pack-normonq $normonq \
    --pack-cross-once 1 --pack-cross-heads $packhead --pack-cross-dim-head 64 \
    --pack-xselfatt $packxselfatt --decoder-attx $decoderattx --decoder-gateatt $decodergate \
    --usevq 1 --vq-codenum $k --vq-dim $vqdim --vq-type km --vq-ema 0.99  --vq-cos 1 \
    --task cg --criterion vq_cross_entropy --vqlamb $vqlamb \
    --lr $lr --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --log-interval 50 \
    --patience 20 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --weight-decay 0.00001 \
    --batch-size $bz --max-update 100000 --no-epoch-checkpoints \
    --max-sentences-valid 512 \
    >>$MODEL/$modelname/train.log 2>&1

#    bash decodecfq.sh $SPLIT $modelname $gpu
done
done

elif [ $mode == 'vqxy' ];then

# valid is sampled from training set
datapath=~/workspace/processdata/cfq-$SPLIT/bin

seed=1
bz=256

g=1
encl=10
decl=6

emadecay=0.99
vqcos=0

vqdim=32

priordp=0.1

vqlamb=1.0

for packnum in 10
do
for k in 128
do
#modelname=cfq${SPLIT}bz${bz}xyv310-6gptg${g}k${k}dim${vqdim}ema${emadecay}pack${packnum}cos${vqcos}lamb${vqlamb}
modelname=see

mkdir -p $MODEL/$modelname

gpu=7

#    --save-interval-updates 1000 --keep-interval-updates 1 \ --keep-last-epochs 5
#--attention-dropout 0.1 --activation-dropout 0.1
    #--finetune-from-model checkpoints/basesd1bz4096lr5e/checkpoint_best.pt \
    #--lr-scheduler fixed --lr 1e-4 \
#    --restore-file checkpoints/priortagxymidg${g}k${k}dim${vqdim}ema${emadecay}pack${packnum}cos$vqcos/checkpoint_best.pt \
#    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer \
#--validate-interval 10
#--eval-bleu --eval-bleu-args '{"beam":1}' --best-checkpoint-metric acc --maximize-best-checkpoint-metric \

CUDA_VISIBLE_DEVICES=$gpu python3 -u fairseq_cli/train.py \
    $datapath \
    --cross-self-attention --no-cross-attention \
    --universal 0 \
    --encoder-embed-dim 256 --encoder-ffn-embed-dim 512 \
    --decoder-embed-dim 256 --decoder-ffn-embed-dim 512 \
    --encoder-attention-heads 4 --decoder-attention-heads 4 \
    --encoder-layers $encl --decoder-layers $decl \
    --seed $seed \
    --arch transformer_vqxyv3_iwslt_de_en \
    --dropout 0.1 \
    --priorbert-train 0 --priorbert-dropout ${priordp} \
    --packnum $packnum --vq-multilinear 0 \
    --vq-tanhffn 0 --vq-codenum $k --vq-group $g --vq-dim $vqdim --vq-type km --vq-ema $emadecay --vq-cos $vqcos \
    --task cg_vq --criterion vqxy_cross_entropy --vqlambda $vqlamb \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --save-dir $MODEL/$modelname \
    --log-interval 50 \
    --patience 20 \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --weight-decay 0.00001 \
    --batch-size $bz --max-update 100000 --no-epoch-checkpoints \
    --max-sentences-valid 512 \
    >$MODEL/$modelname/train.log 2>&1

#    bash decodecfq.sh $SPLIT $modelname $gpu
done
done

fi

#
#CUDA_VISIBLE_DEVICES=$gpu python3 -u $RUN/fairseq_cli/train.py \
#    $datapath \
#    --save-dir $MODEL/$modelname \
#    --change-prob $prob --queue-n $q --rep-layer $layer --rep-minfreq ${minfreq} --knn 1 --mom $mom --soft $soft \
#    --topk $topk --mixfrom $mixfrom --decmix $decmix \
#    --seed 1 \
#    --task translation \
#    --log-interval 100 \
#    --patience 10 \
#    --arch transformer_rep_iwslt_de_en --share-decoder-input-output-embed \
#    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
#    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
#    --dropout 0.3 --weight-decay 0.0001 \
#    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
#    --max-tokens 4096 --max-update 50000 --no-epoch-checkpoints \
#    >>$MODEL/$modelname/train.log 2>&1
 #
#CUDA_VISIBLE_DEVICES=$gpu python -u $RUN/fairseq_cli/generate.py $datapath \
#    --path $MODEL/$modelname/checkpoint_best.pt \
#    --batch-size 256 --beam 5 --lenpen 0.6 --remove-bpe >$MODEL/$modelname/test.log 2>&1

# Evaluate
#RESULT=results/transformer_small_iwslt14_de2en
## RESULT=results/transformer_small_iwslt14_de2en_cutoff
## average last 5 checkpoints
#python scripts/average_checkpoints.py \
#--inputs $RESULT \
#--num-epoch-checkpoints 5 \
#--output $RESULT/checkpoint_last5.pt
## generate results & quick evaluate
#LC_ALL=C.UTF-8 CUDA_VISIBLE_DEVICES=0 python fairseq_cli/generate.py $BIN \
#--path $RESULT/checkpoint_last5.pt \
#--beam 5 --remove-bpe --lenpen 0.5 >> $RESULT/checkpoint_last5.gen
## compound split & re-run evaluate
#bash compound_split_bleu.sh $RESULT/checkpoint_last5.gen
#LC_ALL=C.UTF-8 python fairseq_cli/score.py \
#--sys $RESULT/checkpoint_last5.gen.sys \
#--ref $RESULT/checkpoint_last5.gen.ref
