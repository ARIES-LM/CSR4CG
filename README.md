## Consistency Regularization Training for Compositional Generalization.

### Abstract

Existing neural models have difficulty generalizing to unseen combinations of seen components. To achieve compositional generalization, models are required to consistently interpret (sub)expressions across contexts.
Without modifying model architectures, we improve the capability of Transformer on compositional generalization through consistency regularization training, which promotes representation consistency across samples and prediction consistency for a single sample.
Experimental results on semantic parsing and machine translation benchmarks empirically demonstrate the effectiveness and generality of our method.
In addition, we find that the prediction consistency scores on in-distribution validation sets can be an alternative for evaluating models during training, when commonly-used metrics are not informative.


### Requirements
```
python=3.6.5
pytorch=1.6.0
CUDA=11.0
GPU=Tesla V100
fairseq=1.0.0a0

#install fairseq
cd csrcg_fair010
pip install --editable ./

``` 

### Datasets

- COGS: https://github.com/najoungkim/COGS
- CFQ: https://github.com/google-research/google-research/blob/master/cfq
- CoGnition: https://github.com/yafuly/CoGnition
- OPUS EN-NL: https://github.com/i-machine-think/compositionality_paradox_mt

Preprocessing of COGS and CFQ follows [Dangle](https://github.com/mswellhao/Dangle/tree/main).

Find that different preprocessing strategies are used for COGS and CFQ.
Recent studies adopt variable-free representations [https://github.com/google-research/language/tree/master/language/compgen/csl] and 
and intermediate representations[https://github.com/google-research/language/tree/master/language/compir] for COGS and CFQ, respectively. 

For OPUS EN-NL, we learn joint BPE with 40k merge-operations, which can reduce model size and reproduce the results in the original paper.


### Running scripts
train*.py in csrcg_dangle and csrcg_fair010




```
### Reference:
```bibtex
@inproceedings{csreg,
  title={Consistency Regularization Training for Compositional Generalization},
  author={Yongjing Yin, Jiali Zeng, Yafu Li, Fandong Meng, Jie Zhou, Yue Zhang},
  booktitle={Association for Computational Linguistics (ACL)},
  year={2023}
}
```

