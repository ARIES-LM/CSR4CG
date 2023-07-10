# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from binhex import LINELEN
import math
from re import S
from typing import Any, Dict, List, Optional, Tuple
from unittest import result

import torch
import torch.nn as nn

import torch.nn.functional as F

from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.logging.metrics import reset
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
    GumbelVectorQuantizer,
    KmeansVectorQuantizerVQGAN2, MultiheadAttention
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor
import numpy as np

# added
from fairseq.modules import l0norm

import torch.distributed as dist

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)

def gumbel_sigmoid(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10) -> torch.Tensor:
    uniform = logits.new_empty([2]+list(logits.shape)).uniform_(0,1)

    noise = -((uniform[1] + eps).log() / (uniform[0] + eps).log() + eps).log()
    res = torch.sigmoid((logits + noise) / tau)

    if hard:
        res = ((res > 0.5).type_as(res) - res).detach() + res

    return res


def sigmoid(logits: torch.Tensor, mode: str = "simple", tau: float = 1, eps: float = 1e-10):
    if mode=="simple":
        return torch.sigmoid(logits)
    elif mode in ["soft", "hard"]:
        return gumbel_sigmoid(logits, tau, hard=mode=="hard", eps=eps)
    else:
        assert False, "Invalid sigmoid mode: %s" % mode


def meanshift_torch2(data):
    batch_size = data.size(0)
    n = len(data)
    X = torch.from_numpy(np.copy(data)).cuda()
    for _ in range(5):
        for i in range(0, n, batch_size):
            s = slice(i, min(n, i + batch_size))
            weight = gaussian(distance_batch(X, X[s]), 2.5)
            num = (weight[:, :, None] * X).sum(dim=1)
            X[s] = num / weight.sum(1)[:, None]
    return X


@register_model("transformer_hybrid")
class TransformerModel(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        # fmt: off

        def moses_subword(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'subword_nmt',
            }

        def moses_fastbpe(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'fastbpe',
            }

        def spm(path):
            return {
                'path': path,
                'bpe': 'sentencepiece',
                'tokenizer': 'space',
            }

        return {
            'transformer.wmt14.en-fr': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2'),
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz'),
            'transformer.wmt19.en-de': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.en-ru': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz'),
            'transformer.wmt19.de-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.ru-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz'),
            'transformer.wmt19.en-de.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.en-ru.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz'),
            'transformer.wmt19.de-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.ru-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz'),
            'transformer.wmt20.en-ta': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-ta.single.tar.gz'),
            'transformer.wmt20.en-iu.news': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-iu.news.single.tar.gz'),
            'transformer.wmt20.en-iu.nh': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.en-iu.nh.single.tar.gz'),
            'transformer.wmt20.ta-en': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.ta-en.single.tar.gz'),
            'transformer.wmt20.iu-en.news': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.iu-en.news.single.tar.gz'),
            'transformer.wmt20.iu-en.nh': spm('https://dl.fbaipublicfiles.com/fairseq/models/wmt20.iu-en.nh.single.tar.gz'),
            'transformer.flores101.mm100.615M': spm('https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_615M.tar.gz'),
            'transformer.flores101.mm100.175M': spm('https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_175M.tar.gz'),
        }
        # fmt: on

    def __init__(self, args, encoder, decoder, ims_encoder, ims_decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True

        # added
        self.ims_encoder = ims_encoder
        self.ims_decoder = ims_decoder

        if self.args.train_stage == 1:
            self.encoder.requires_grad_(False)
            self.decoder.requires_grad_(False)
        elif self.args.train_stage == 2:
            self.ims_encoder.requires_grad_(False)
            self.ims_decoder.requires_grad_(False)
            self.encoder.ims_encoder = self.ims_encoder

        if args.cllamb > 0:
            emddim = args.encoder_embed_dim
            self.proj = nn.Sequential(nn.Linear(emddim, emddim * 2), nn.ReLU(),
                                    nn.Linear(emddim * 2, 128))

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension (extra linear layer '
                                 'if different from decoder embed dim')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        parser.add_argument('--checkpoint-activations', action='store_true',
                            help='checkpoint activations at each layer, which saves GPU '
                                 'memory usage at the cost of some additional compute')
        parser.add_argument('--offload-activations', action='store_true',
                            help='checkpoint activations at each layer, then save to gpu. Sets --checkpoint-activations.')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq', type=float, metavar='D', default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size', type=int, metavar='D', default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument('--quant-noise-scalar', type=float, metavar='D', default=0,
                            help='scalar quantization noise and scalar quantization at training time')
        # args for Fully Sharded Data Parallel (FSDP) training
        parser.add_argument(
            '--min-params-to-wrap', type=int, metavar='D', default=DEFAULT_MIN_PARAMS_TO_WRAP,
            help=(
                'minimum number of params for a layer to be wrapped with FSDP() when '
                'training with --ddp-backend=fully_sharded. Smaller values will '
                'improve memory efficiency, but may make torch.distributed '
                'communication less efficient due to smaller input sizes. This option '
                'is set to 0 (i.e., always wrap) when --checkpoint-activations or '
                '--offload-activations are passed.'
            )
        )
        # fmt: on
        # added
        parser.add_argument('--universal', type=int, default=0, help='share weights between layers')
        parser.add_argument('--emdinit', default=None, )
        parser.add_argument('--scaledown', default=0, help='down:')

        # for vq
        parser.add_argument('--usevq', type=int, default=0)
        parser.add_argument('--vq-type', type=str, default="km")
        parser.add_argument('--vq-codenum', type=int, default=128)
        parser.add_argument('--vq-group', type=int, default=2)
        parser.add_argument('--vq-decay', type=float, default=0.9999, help='gumbel decay')
        parser.add_argument('--vq-dim', type=int, default=128)
        parser.add_argument('--vq-tanhffn', type=int, default=0)
        parser.add_argument('--vq-multilinear', type=int, default=0)
        parser.add_argument('--vq-cos', type=int, default=0)
        parser.add_argument('--vq-ema', type=float, default=0.0)

        # for pack
        parser.add_argument('--packnum', type=int, default=5)
        parser.add_argument('--pack-cross-once', type=int, default=0)
        parser.add_argument('--pack-cross-heads', type=int, default=1)
        parser.add_argument('--pack-cross-dim-head', type=int, default=64)
        parser.add_argument('--pack-token-positional-embeddings', type=int, default=1)
        parser.add_argument('--pack-normonq', type=int, default=0)
        parser.add_argument('--pack-xselfatt', type=int, default=0, help='keep original x self-att')

        parser.add_argument('--dynpacknum', type=int, default=0)
        parser.add_argument('--addseqrep', type=int, default=0)

        parser.add_argument('--decoder-attx', type=int, default=0)
        parser.add_argument('--decoder-gateatt', type=int, default=0)

        #
        parser.add_argument('--train-stage', type=int, default=1)
        parser.add_argument('--pack-iters', type=int, default=3)

        parser.add_argument('--l0warmb', type=int, default=0)
        parser.add_argument('--l0warme', type=int, default=0)

        # cl
        parser.add_argument("--cltau", default=0.07, type=float,metavar="D",)
        parser.add_argument("--cllamb", default=0.0, type=float,metavar="D",)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)

        # added
        encoder_embed_tokens2 = cls.build_embedding(
            args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
        )
        ims_encoder = cls.build_imsencoder(args, src_dict, encoder_embed_tokens2)
        ims_decoder = cls.build_imsdecoder(args, tgt_dict, encoder_embed_tokens2)

        if not args.share_all_embeddings:
            min_params_to_wrap = getattr(
                args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=min_params_to_wrap)

        return cls(args, encoder, decoder, ims_encoder, ims_decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx, args.emdinit)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoder(args, src_dict, embed_tokens)

    # added
    @classmethod
    def build_imsencoder(cls, args, src_dict, embed_tokens):
        return IMSEncoder(args, src_dict, embed_tokens)

    @classmethod
    def build_imsdecoder(cls, args, tgt_dict, embed_tokens):
        return IMSDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    # added
    def _sent_rep(self, h, mask):
        if mask.dtype == torch.bool:
            h = h.masked_fill(mask.unsqueeze(-1), 0)
            h = h.sum(1) / (~mask).sum(1, True)
        else:
            # print('ok')
            h = h * mask.unsqueeze(-1)
            h = h.sum(1) / (mask.sum(1, True) + 1e-8)

        # exit()

        return h

    def train_ims(self, src_tokens, src_lengths, prev_output_tokens_x, target_tokens, tgt_lengths, prev_output_tokens):
        # print(prev_output_tokens_x.size(), prev_output_tokens.size())
        # print(src_tokens)
        # print(prev_output_tokens_x)
        #
        # print()
        # print(target_tokens)
        # print(prev_output_tokens)

        # for src
        encoder_out_src = self.ims_encoder(
            src_tokens, src_lengths=src_lengths,
        )

        logtis4x, extra4x = self.ims_decoder(
            prev_output_tokens_x,
            encoder_out=encoder_out_src,
            features_only=False,
            src_lengths=src_lengths,
        )

        # for tgt
        encoder_out = self.ims_encoder(
            target_tokens, src_lengths=tgt_lengths,
        )
        logtis4y, extra4y = self.ims_decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=False,
            src_lengths=tgt_lengths,
        )

        extra4y['logits4x'] = logtis4x

        extra4y['l0loss'] = (extra4x['l0loss'] + extra4y['l0loss']) / 2.0

        if self.args.cllamb > 0 and self.training:
            # B T
            src_rep = encoder_out_src['encoder_out'][0].transpose(0, 1)
            srcims_rep = self._sent_rep(src_rep, encoder_out_src['encoder_padding_mask'][0])

            tgt_rep = encoder_out['encoder_out'][0].transpose(0, 1)
            tgtims_rep = self._sent_rep(tgt_rep, encoder_out['encoder_padding_mask'][0])
            # B H
            z1 = self.proj(srcims_rep)
            z2 = self.proj(tgtims_rep)

            # gather for mgpu
            # if set, set batch-size instead of max-tokens

            # Dummy vectors for allgather
            # z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            # z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            # # Allgather
            # dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            # dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())
            #
            # # Since allgather results do not have gradients, we replace the
            # # current process's corresponding embeddings with original tensors
            # z1_list[dist.get_rank()] = z1
            # z2_list[dist.get_rank()] = z2
            # # Get full batch embeddings: (bs x N, hidden)
            # z1 = torch.cat(z1_list, 0)
            # z2 = torch.cat(z2_list, 0)

            logits = F.cosine_similarity(z1.unsqueeze(1), z2.unsqueeze(0), dim=-1)
            logits = logits / self.args.cltau
            label = torch.arange(logits.size(0)).to(logits.device)
            clloss = F.cross_entropy(logits, label)
            clloss = clloss * self.args.cllamb
            extra4y['clloss'] = clloss

        return logtis4y, extra4y

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        **kwargs
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """

        # added
        if kwargs['stage'] == 1:
            prev_output_tokens_x = torch.full_like(src_tokens, prev_output_tokens[0,0])
            prev_output_tokens_x[:, 1:] = src_tokens[:, :-1]

            tgt_lengths = (kwargs['target'] != self.decoder.padding_idx).long().sum(-1)

            stage1_out = self.train_ims(src_tokens, src_lengths, prev_output_tokens_x,
                                    kwargs['target'], tgt_lengths, prev_output_tokens)
            return stage1_out

        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )

        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )

        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


# added
class PackMultiheadAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
            self,
            embed_dim,
            num_heads,
            head_dim,
            kdim=None,
            vdim=None,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            self_attention=False,
            encoder_decoder_attention=False,
            q_noise=0.0,
            qn_block_size=8,
            norm_on_q=0
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # self.kdim = kdim if kdim is not None else embed_dim
        # self.vdim = vdim if vdim is not None else embed_dim

        # self.kdim = kdim if kdim is not None else embed_dim
        # self.vdim = vdim if vdim is not None else embed_dim

        self.qkv_same_dim = embed_dim == head_dim * num_heads

        self.num_heads = num_heads

        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )

        self.head_dim = head_dim
        # assert (
        #         self.head_dim * num_heads == self.embed_dim
        # ), "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention

        assert not self.self_attention or self.qkv_same_dim, (
            "Self-attention requires query, key and " "value to be of the same size"
        )

        # added
        innerdim = head_dim * num_heads

        self.k_proj = apply_quant_noise_(
            nn.Linear(embed_dim, innerdim, bias=bias), q_noise, qn_block_size
        )
        self.v_proj = apply_quant_noise_(
            nn.Linear(embed_dim, innerdim, bias=bias), q_noise, qn_block_size
        )
        self.q_proj = apply_quant_noise_(
            nn.Linear(embed_dim, innerdim, bias=bias), q_noise, qn_block_size
        )

        self.out_proj = apply_quant_noise_(
            nn.Linear(innerdim, embed_dim, bias=bias), q_noise, qn_block_size
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False
        # added
        self.norm_on_q = norm_on_q

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qkv_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
            self,
            query,
            key: Optional[Tensor],
            value: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            need_weights: bool = True,
            static_kv: bool = False,
            attn_mask: Optional[Tensor] = None,
            before_softmax: bool = False,
            need_head_weights: bool = False,
            **kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        # print('query.size(), key.size()', query.size(), key.size())

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)

        q *= self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0), 1),
                    ],
                    dim=1,
                )

        q = (
            q.contiguous()
                .view(tgt_len, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                    .view(-1, bsz * self.num_heads, self.head_dim)
                    .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                    .view(-1, bsz * self.num_heads, self.head_dim)
                    .transpose(0, 1)
            )

        assert k is not None
        # added
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        # print(attn_weights.size(), tgt_len, src_len)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)

            attn_weights += attn_mask

        # added
        # for normal attention, padding mask is applied to the energy
        if key_padding_mask is not None and not self.norm_on_q:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if not is_tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                    float("-inf"),
                )
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        # added
        if self.norm_on_q:
            attn_weights_float = utils.softmax(
                attn_weights, dim=1, onnx_trace=self.onnx_trace
            )
        else:
            attn_weights_float = utils.softmax(
                attn_weights, dim=-1, onnx_trace=self.onnx_trace
            )

        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None

        # added
        if self.norm_on_q and key_padding_mask is not None:
            # padding mask is applied to hidden of v
            # print(v.size(), key_padding_mask.size())
            v = v.view(bsz, self.num_heads, -1, self.head_dim)
            v = v.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool), 0.0)
            v = v.view(bsz*self.num_heads, -1, self.head_dim)

        attn = torch.bmm(attn_probs, v)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        # added: should first project
        if self.num_heads == 1:
            attn = self.out_proj(attn)

        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)

        # added
        if self.num_heads > 1:
            attn = self.out_proj(attn)

        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights


    def apply_sparse_mask(self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim: 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim:]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                                                           dim: 2 * dim
                                                           ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim:]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value


# added
class CrossAttLayer(nn.Module):

    def __init__(self, args, cross_att=1):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8) or 8
        export = getattr(args, "export", False)

        # added
        self.cross_att = cross_att
        if cross_att:
            self.cross_attn = self.build_cross_attention(self.embed_dim, args)
            self.cross_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.xselfatt = args.pack_xselfatt
        if args.pack_xselfatt:
            self.self_attn_x = self.build_self_attention(self.embed_dim, args)
            self.self_attn_x_layer_norm = LayerNorm(self.embed_dim, export=export)

            self.fc1_x = self.build_fc1(
                self.embed_dim,
                args.encoder_ffn_embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.fc2_x = self.build_fc2(
                args.encoder_ffn_embed_dim,
                self.embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.ffnx_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu") or "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return apply_quant_noise_(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return apply_quant_noise_(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    # added
    def build_cross_attention(self, embed_dim, args):
        return PackMultiheadAttention(
            embed_dim,
            args.pack_cross_heads,
            args.pack_cross_dim_head,
            dropout=args.attention_dropout,
            self_attention=False,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            norm_on_q=args.pack_normonq
        )

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        packemb,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
        pack_padding_mask=None
    ):
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        if self.xselfatt:
            residual = x
            if self.normalize_before:
                x = self.self_attn_x_layer_norm(x)
            x, _ = self.self_attn_x(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.self_attn_x_layer_norm(x)

            residual = x
            if self.normalize_before:
                x = self.ffnx_layer_norm(x)
            x = self.activation_fn(self.fc1_x(x))
            x = self.activation_dropout_module(x)
            x = self.fc2_x(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.ffnx_layer_norm(x)

        # 1
        if self.cross_att:
            residual = packemb
            if self.normalize_before:
                packemb = self.cross_attn_layer_norm(packemb)

            packemb, _ = self.cross_attn(
                query=packemb,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
                attn_mask=attn_mask,
            )
            packemb = self.dropout_module(packemb)
            packemb = self.residual_connection(packemb, residual)
            if not self.normalize_before:
                packemb = self.cross_attn_layer_norm(packemb)

        # 2
        residual = packemb
        if self.normalize_before:
            packemb = self.self_attn_layer_norm(packemb)

        packemb, _ = self.self_attn(
            query=packemb,
            key=packemb,
            value=packemb,
            key_padding_mask=pack_padding_mask,
        )
        packemb = self.dropout_module(packemb)
        packemb = self.residual_connection(packemb, residual)
        if not self.normalize_before:
            packemb = self.self_attn_layer_norm(packemb)

        # 3
        residual = packemb
        if self.normalize_before:
            packemb = self.final_layer_norm(packemb)
        packemb = self.activation_fn(self.fc1(packemb))
        packemb = self.activation_dropout_module(packemb)
        packemb = self.fc2(packemb)
        packemb = self.dropout_module(packemb)
        packemb = self.residual_connection(packemb, residual)
        if not self.normalize_before:
            packemb = self.final_layer_norm(packemb)

        return packemb, x

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                # in_proj_weight used to be q + k + v with same dimensions
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim : 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim :]

                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    dim = int(state_dict[k].shape[0] / 3)
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][
                        dim : 2 * dim
                    ]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim :]

                    keys_to_remove.append(prefix + "in_proj_bias")

        for k in keys_to_remove:
            del state_dict[k]

        for key, value in items_to_add.items():
            state_dict[key] = value


# added
class PackEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )
        export = getattr(args, "export", False)
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim, export=export)
        else:
            self.layernorm_embedding = None

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        # added
        if args.universal:
            self.layers = nn.ModuleList([self.build_encoder_layer(args)] * args.pack_iters)
        else:
            # if self.encoder_layerdrop > 0.0:
            #     self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
            # else:
            self.layers = nn.ModuleList([])
            self.layers.extend(
                [self.build_encoder_layer(args, i) for i in range(args.pack_iters)]
             )

        # self.layers = nn.ModuleList([self.build_encoder_layer(args)])
        self.num_layers = len(self.layers)

        self.pack_iters = args.pack_iters

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim, export=export)
        else:
            self.layer_norm = None

        # added
        packembs = torch.empty(args.packnum, embed_dim)
        nn.init.normal_(packembs, mean=0, std=embed_dim ** -0.5)
        self.register_parameter("pack_embs", nn.Parameter(packembs))

        # self.embln = LayerNorm(embed_dim, export=export)
        # self.embln2 = LayerNorm(embed_dim, export=export)
        # self.packln = LayerNorm(embed_dim, export=export)
        # self.hardffn1 = nn.Linear(embed_dim*2, embed_dim)
        # self.hardffn2 = nn.Linear(embed_dim, embed_dim)
        # self.hardffn3 = nn.Linear(embed_dim, 1)
        # self.embffn1 = nn.Linear(embed_dim, embed_dim*2)
        # self.embffn2 = nn.Linear(embed_dim*2, embed_dim)
        # self.embffn3 = nn.Linear(embed_dim*2, embed_dim)


        vqtype = args.vq_type
        self.vqtype = vqtype
        usevq = args.usevq
        self.usevq = usevq
        self.multilinear = 0
        if usevq and vqtype == 'gumbel':
            vq_dim = args.vq_dim
            self.input_quantizer = GumbelVectorQuantizer(
                dim=embed_dim,
                num_vars=args.vq_codenum,
                temp=(2, 0.5, args.vq_decay),
                groups=args.vq_group,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
                activation=nn.ReLU(),
                weight_proj_depth=2,
                weight_proj_factor=1,
            )

            self.vqfroml = args.vq_froml
            self.project_inps = nn.ModuleList([])
            self.vq_layer_norms = nn.ModuleList([])
            for _ in range(self.num_layers - args.vq_froml + 1):
                self.project_inps.append(nn.Linear(vq_dim, embed_dim))
                self.vq_layer_norms.append(LayerNorm(embed_dim, export=export))
        elif usevq and vqtype == 'km':
            self.vq_tanhffn = args.vq_tanhffn
            if args.vq_tanhffn:
                self.encfc1 = Linear(embed_dim, embed_dim*2)
                self.encfc2 = Linear(embed_dim*2, embed_dim)

            vq_dim = args.vq_dim
            self.input_quantizer = KmeansVectorQuantizerVQGAN2(
                dim=embed_dim,
                num_vars=args.vq_codenum,
                groups=args.vq_group,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
                ema=args.vq_ema,
                cosine=args.vq_cos
            )

            if args.vq_dim < args.encoder_embed_dim:
                self.proj2z = Linear(args.encoder_embed_dim, args.vq_dim, False)
                self.proj2x = Linear(args.vq_dim, args.encoder_embed_dim, False)

        self.register_buffer('codeuse', torch.zeros(args.vq_codenum))

    def build_encoder_layer(self, args, i=None):
        # added
        if self.args.pack_cross_once:
            if i == 0:
                layer = CrossAttLayer(args, cross_att=1)
            else:
                layer = CrossAttLayer(args, cross_att=0)
        else:
            layer = CrossAttLayer(args, cross_att=1)

        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint
            else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    # added
    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    # added
    def gumbel_quantize(self, x, mask):
        residual = x
        x = x.transpose(0, 1)
        
        q = self.input_quantizer(x, produce_targets=False, mask=encoder_padding_mask)

        features = q["x"]
        num_vars = q["num_vars"]
        # code_ppl = q["code_perplexity"]
        prob_ppl = q["prob_perplexity"]
        curr_temp = q["temp"]

        x = self.project_inps[ix - (self.vqfroml - 1)](features)
        x = x.transpose(0, 1)

        x = residual + self.vqdp(x)
        x = self.vq_layer_norms[ix - (self.vqfroml - 1)](x)
        
        diverloss += (num_vars - prob_ppl) / num_vars

    def hardsigmoid(self, x):
        logits = self.hardffn1(x)

        m = sigmoid(logits, 'hard')
        return m

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None, addposition=True
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)

        x = embed = self.embed_scale * token_embedding

        if self.embed_positions is not None:
            # added
            if addposition:
                x = embed + self.embed_positions(src_tokens)

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def kmvq(self, x):
        # tanh proj
        if self.vq_tanhffn:
            x = torch.tanh(self.encfc1(x))
            x = self.encfc2(x)

        if self.args.vq_dim < self.args.encoder_embed_dim:
            if self.multilinear:
                headdim = x.size(-1) // self.args.vq_group
                x = torch.cat([self.proj2z[gix](xi) for gix, xi in enumerate(x.split(headdim, dim=-1))], -1)
            else:
                x = self.proj2z(x)

        # -> * H
        psize, bsize, hsize = x.size()

        unquan_x = x.view(-1, x.size(-1))

        q = self.input_quantizer(unquan_x)
        quan_x = q["x"]
        idx = q["idx"]

        # P B G
        idx = idx.view(psize, bsize, -1)

        # B PG
        # idx4print = idx.transpose(0, 1).reshape(idx.size(0), -1)
        # print(idx4print.view(-1))

        kmloss = q["kmeans_loss"]
        # codeppl = q["code_perplexity"]

        if self.training:
            self.codeuse = self.codeuse.float()
            self.codeuse += q["codeuse"].float().detach()

        # P B Z
        quan_x = quan_x.reshape(psize, bsize, -1)

        if self.args.vq_dim < self.args.encoder_embed_dim:
            # if self.multilinear:
            #     headdim = quan_x.size(-1) // self.args.vq_group
            #     x = torch.cat([self.proj2x[gix](xi) for gix, xi in enumerate(quan_x.split(headdim, dim=-1))], -1)
            # else:
            x = self.proj2x(quan_x)
        else:
            x = quan_x

        return {'x': x, 'loss': kmloss}

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(
            src_tokens, src_lengths, return_all_hiddens, token_embeddings
        )

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B H
        if self.args.addseqrep:
            seq_rep = x.sum(1) / src_lengths.unsqueeze(-1)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        # encoder layers
        # added
        # B P C
        # print(src_lengths)
        if self.args.dynpacknum:
            dypacknum = torch.log(src_lengths.float()) / math.log(1.2)
            minlen = torch.min(dypacknum, src_lengths.float())
            minlen = torch.min(minlen, minlen.new_tensor([self.args.packnum]))
            minlen = minlen.long()

        packemb = self.pack_embs
        packemb = packemb.unsqueeze(0).expand(src_tokens.size(0), -1, -1)

        packemb = packemb * self.embed_scale
        packemb = self.dropout_module(packemb)

        if self.args.addseqrep:
            packemb = packemb + seq_rep.unsqueeze(1)

        # P B C
        packemb = packemb.transpose(0, 1)

        if return_all_hiddens:
            encoder_states.append(packemb)

        # B P
        pack_padding_mask = encoder_padding_mask.new_zeros(src_tokens.size(0), packemb.size(0))

        if self.args.dynpacknum:
            for ib in range(pack_padding_mask.size(0)):
                pack_padding_mask[ib, minlen[ib]:] = 1

        # for ix, layer in enumerate(self.layers):
        for ix in range(self.pack_iters):
            packemb, x = self.layers[ix](
                    packemb, x, encoder_padding_mask=encoder_padding_mask if has_pads else None,
                pack_padding_mask=pack_padding_mask
                )

            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(packemb)

        if self.layer_norm is not None:
            packemb = self.layer_norm(packemb)

        if self.usevq:
            vqoutput = self.kmvq(packemb)
            packemb = vqoutput['x']
            vqloss = vqoutput['loss']
        else:
            vqloss = 0

        # vocab
        '''
        src_vocab = []
        for l in src_tokens:
            l = set(l.tolist())
            src_vocab.append(src_tokens.new_tensor(list(l)))

        src_vocab = torch.nn.utils.rnn.pad_sequence(src_vocab, batch_first=True, padding_value=self.padding_idx)
        src_vocab_emb, _ = self.forward_embedding(src_vocab, addposition=False)

        padding_mask = src_vocab.eq(self.padding_idx)
        '''

        # seq_rep = seq_rep.unsqueeze(0).expand_as(packemb)
        # select_condi = torch.cat((packemb, seq_rep), -1)
        # # P B
        # pack_padding_mask = self.hardsigmoid(select_condi)
        # pack_padding_mask = pack_padding_mask.T.bool()

        # print('mean', x[~encoder_padding_mask.T].mean().item(),
        #       encoder_embedding[~encoder_padding_mask].mean().item())
        # token_embedding = self.embed_tokens(src_tokens)
        # print('mean', token_embedding[~encoder_padding_mask].mean().item())

        # token_embedding = self.embln(token_embedding)
        # print(token_embedding[~encoder_padding_mask].mean())

        # encoder_embedding = encoder_embedding + self.embed_positions(src_tokens)
        # encoder_embedding = self.dropout_module(encoder_embedding)

        # use encoder output
        encoder_embedding = x.transpose(0, 1)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [packemb],  # T x B x C
            "encoder_padding_mask": [pack_padding_mask],  # B x P
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "src_emb_padding_mask": [encoder_padding_mask], # B T
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "vqloss": [vqloss]
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]

        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]

        # added
        if len(encoder_out["src_emb_padding_mask"]) == 0:
            new_src_emb_padding_mask = []
        else:
            new_src_emb_padding_mask = [
                encoder_out["src_emb_padding_mask"][0].index_select(0, new_order)
            ]

        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "src_emb_padding_mask": new_src_emb_padding_mask,  # B N
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class IMSLearner(FairseqEncoder):
    def __init__(self, args,):
        super().__init__(None)
        self.args = args

        hdim = args.encoder_embed_dim
        self.select_ln = nn.Linear(hdim, 1)

        self.glayers = nn.ModuleList([TransformerGraphLayer(args) for _ in range(2)])

        edge_num = 100
        self.edge_ln = nn.Linear(hdim*2, edge_num)

        head_dim = hdim // args.encoder_attention_heads
        self.head_dim = head_dim
        self.edge_embs = nn.Parameter(torch.FloatTensor(edge_num, head_dim))
        nn.init.normal_(self.edge_embs)

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def token_select(self, source_memory, source_mask):
        '''
        :param source_memory:
        :param source_mask: 1 means token, 0 means pad
        :return:
        '''
        # source_pruning: log alpha_i = x_i w^T
        source_pruning = self.select_ln(source_memory)

        # if self.training:  # training
        source_memory, l0_mask = l0norm.var_train((source_memory, source_pruning))
        # 0 ~ 1
        # print(l0_mask[0])

        l0_norm_loss = l0norm.l0_norm(source_pruning).squeeze(-1)
        l0_norm_loss = torch.sum(l0_norm_loss * source_mask, -1) / torch.sum(source_mask, -1)
        l0_norm_loss = torch.mean(l0_norm_loss)

        # B T 1
        # print(l0_mask.size())
        # print(l0_mask)
        l0_warn_start = self.args.l0warmb
        l0_warm_end = self.args.l0warme
        if self.training:
            l0_norm_loss = l0norm.l0_regularization_loss(
            l0_norm_loss,
            step=self.num_updates,
            reg_scalar=0.2,
            start_reg_ramp_up=l0_warn_start,
            end_reg_ramp_up=l0_warm_end,
            warm_up=l0_warm_end > l0_warn_start,
            )
        else:
            l0_norm_loss = source_memory.new_zeros(1)

        l0_mask = l0_mask.squeeze(-1).bool()
        new_source_mask = l0_mask & source_mask

        if self.num_updates % 1000 == 0:
            dropnum = source_mask.sum(-1) - new_source_mask.sum(-1)
            print('drop rate', dropnum[:50].float() / source_mask.sum(-1)[:50].float())

        # force the model to only attend to unmasked position
        # B T
        source_mask = new_source_mask.to(source_memory.dtype)

        count_mask = source_mask
        # else:  # evaluation
        #     source_memory, l0_mask = l0norm.var_eval((source_memory, source_pruning))
        #     l0_norm_loss = 0.0
        #
        #     source_memory, source_mask, count_mask = l0norm.extract_encodes(source_memory, source_mask, l0_mask)
            # count_mask = tf.expand_dims(tf.expand_dims(count_mask, 1), 1)

        return source_memory, source_mask, count_mask, l0_norm_loss

    def graph_encoding(self, h, mask, ):
        # B N N H
        tsize = h.size(1)

        # B N H, B N H E
        hh = torch.cat((h.unsqueeze(2).expand(-1, -1, tsize, -1), h.unsqueeze(1).expand(-1, tsize, -1, -1)), -1)

        # B N N E
        # todo
        edge_prob = F.gumbel_softmax(self.edge_ln(hh).float(), tau=1.0, hard=False).type_as(h)
        # 1 1 E H
        edge_emb = self.edge_embs.unsqueeze(0).unsqueeze(1)
        # B N N H
        edge_emb = torch.matmul(edge_prob, edge_emb)
        # B*N H N
        edge_emb = edge_emb.view(-1, tsize, self.head_dim).transpose(1, 2)

        # !
        h = h.transpose(0, 1)
        for l in self.glayers:
            h = l(h, mask, edge_emb=edge_emb)

        return h

    def forward(self, out):
        '''
        node and edge selection, graph encoding
        :param h: T B H
        :param padding_mask: B T, 1 means pad
        :return:
        '''
        # b t h
        h = out['encoder_out'][0].transpose(0, 1)
        mask = ~out['encoder_padding_mask'][0]

        # node selection
        # B T
        h, source_mask, count_mask, l0_norm_loss = self.token_select(h, mask)

        # print(source_mask[0])
        # print(l0_norm_loss)
        graph_h = self.graph_encoding(h, source_mask)
        # print(graph_h.size())
        # exit('ok')
        return {'h': graph_h, 'mask': source_mask, 'l0loss': l0_norm_loss}


class TransformerEncoder(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = (
            PositionalEmbedding(
                args.max_source_positions,
                embed_dim,
                self.padding_idx,
                learned=args.encoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )
        export = getattr(args, "export", False)
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim, export=export)
        else:
            self.layernorm_embedding = None

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        # added
        if args.universal:
            self.layers = nn.ModuleList([self.build_encoder_layer(args)] * args.encoder_layers)
        else:
            if self.encoder_layerdrop > 0.0:
                self.layers = LayerDropModuleList(p=self.encoder_layerdrop)
            else:
                self.layers = nn.ModuleList([])

            self.layers.extend(
                [self.build_encoder_layer(args, i) for i in range(args.encoder_layers)]
            )

        self.num_layers = len(self.layers)

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim, export=export)
        else:
            self.layer_norm = None

        # added
        self.ims_encoder = None

    def build_encoder_layer(self, args, i=None):
        layer = TransformerEncoderLayer(args)

        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint
            else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward_embedding(
        self, src_tokens, token_embedding: Optional[torch.Tensor] = None
    ):
        # embed tokens and positions
        if token_embedding is None:
            token_embedding = self.embed_tokens(src_tokens)

        x = embed = self.embed_scale * token_embedding

        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)
        if self.quant_noise is not None:
            x = self.quant_noise(x)
        return x, embed

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        return self.forward_scriptable(
            src_tokens, src_lengths, return_all_hiddens, token_embeddings
        )

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for ix, layer in enumerate(self.layers):
            x = layer(
                    x, encoder_padding_mask=encoder_padding_mask if has_pads else None,
                )

            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # added
        if self.ims_encoder is not None:
            # stage 2

            self.ims_encoder.eval()

            with torch.no_grad():
                ims_encoder_out = self.ims_encoder(
                    src_tokens, src_lengths=src_lengths,
                    return_all_hiddens=return_all_hiddens,
                )

            ims_x = ims_encoder_out['encoder_out'][0]
            ims_x = F.dropout(ims_x, p=self.args.dropout, training=self.training)

            x = torch.cat((ims_x, x), 0)

            ims_mask = ims_encoder_out['encoder_padding_mask'][0]
            # float -> bool
            ims_mask = ims_mask.bool()
            ims_mask = ~ims_mask

            encoder_padding_mask = torch.cat((ims_mask, encoder_padding_mask), 1)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions)

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                print("deleting {0}".format(weights_key))
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)
        for i in range(self.num_layers):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(
                state_dict, "{}.layers.{}".format(name, i)
            )

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class IMSEncoder(TransformerEncoder):
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        self.imslearner = IMSLearner(args)

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings

        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        for ix, layer in enumerate(self.layers):
            x = layer(
                    x, encoder_padding_mask=encoder_padding_mask if has_pads else None,
                )

            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        out = {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

        # added
        new_out = self.imslearner(out)

        assert x.size() == new_out['h'].size()

        out['encoder_out'][0] = new_out['h']
        out['encoder_padding_mask'][0] = new_out['mask']
        out['l0loss'] = new_out['l0loss']

        return out


class TransformerDecoderLayerv2(TransformerDecoderLayer):
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttentionv2(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )
        export = getattr(args, "export", False)
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim, export=export)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        # added
        if args.universal:
            self.layers = nn.ModuleList([self.build_decoder_layer(args, no_encoder_attn)]*args.decoder_layers)
        else:
            if self.decoder_layerdrop > 0.0:
                self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
            else:
                self.layers = nn.ModuleList([])
            self.layers.extend(
                [
                    self.build_decoder_layer(args, no_encoder_attn)
                    for _ in range(args.decoder_layers)
                ]
            )

        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim, export=export)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(args, dictionary, embed_tokens)

    def build_output_projection(self, args, dictionary, embed_tokens):
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )
        num_base_layers = getattr(args, "base_layers", 0)
        for i in range(num_base_layers):
            self.layers.insert(
                ((i + 1) * args.decoder_layers) // (num_base_layers + 1),
                BaseLayer(args),
            )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = TransformerDecoderLayer(args, no_encoder_attn)

        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint
            else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        **kwargs
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        # added
        if self.training:
            # extra['srch'] = encoder_out['encoder_out'][0]
            extra['tgth'] = x

        if 'l0loss' in encoder_out:
            extra['l0loss'] = encoder_out['l0loss']
        else:
            extra['l0loss'] = x.new_zeros([1])

        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """

        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        # inner_states: List[Optional[Tensor]] = [x]

        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )

            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn] }

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


class IMSDecoder(TransformerDecoder):
    '''
    cross-att -> float, 0/1 mask
    layer->3
    '''

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False, output_projection=None,):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn, output_projection,)
        # added
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, no_encoder_attn)
                for _ in range(3)
            ]
        )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        # added
        layer = TransformerDecoderLayerv2(args, no_encoder_attn)

        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint
            else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer


'''
class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        export = getattr(args, "export", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        # added
        if no_encoder_attn:
            self.pack_attn = None
            self.pack_attn_layer_norm = None
        else:
            self.pack_attn = self.build_encoder_attention(self.embed_dim, args)
            self.pack_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

            if args.decoder_gateatt:
                self.gate = nn.Sequential(Linear(self.embed_dim*2, self.embed_dim), nn.Tanh(),
                                          Linear(self.embed_dim, self.embed_dim), nn.Sigmoid())
            else:
                self.gate = None

        if args.decoder_attx:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
        else:
            self.encoder_attn = None

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return apply_quant_noise_(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return apply_quant_noise_(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        **kwargs
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)

        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)

        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        # print(x.size(), self_attn_padding_mask.size())

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # added
        if self.pack_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.pack_attn_layer_norm(x)

            # if prev_pack_state is not None:
            #     prev_key, prev_value = prev_attn_state[:2]
            #     saved_state: Dict[str, Optional[Tensor]] = {
            #         "prev_key": prev_key,
            #         "prev_value": prev_value,
            #     }
            #     if len(prev_attn_state) >= 3:
            #         saved_state["prev_key_padding_mask"] = prev_attn_state[2]
            #     assert incremental_state is not None
            #     self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.pack_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )

            if self.gate is not None:
                gate_input = torch.cat((x, residual), -1)
                gate = self.gate(gate_input)
                x = x * gate

            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.pack_attn_layer_norm(x)

        if self.encoder_attn is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            # added
            src_emb = kwargs['src_emb'].transpose(0, 1)
            src_emb_padding_mask = kwargs['src_emb_mask']

            x, attn = self.encoder_attn(
                query=x,
                key=src_emb, #
                value=src_emb, #
                key_padding_mask=src_emb_padding_mask, #
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state

        return x, attn, None

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn


class TransformerDecoder(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.args = args
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        if not args.adaptive_input and args.quant_noise_pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                args.quant_noise_pq,
                args.quant_noise_pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=args.decoder_learned_pos,
            )
            if not args.no_token_positional_embeddings
            else None
        )
        export = getattr(args, "export", False)
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim, export=export)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        # added
        if args.universal:
            self.layers = nn.ModuleList([self.build_decoder_layer(args, no_encoder_attn)]*args.decoder_layers)
        else:
            if self.decoder_layerdrop > 0.0:
                self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
            else:
                self.layers = nn.ModuleList([])
            self.layers.extend(
                [
                    self.build_decoder_layer(args, no_encoder_attn)
                    for _ in range(args.decoder_layers)
                ]
            )

        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(embed_dim, export=export)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(args, dictionary, embed_tokens)


    def build_output_projection(self, args, dictionary, embed_tokens):
        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )
        num_base_layers = getattr(args, "base_layers", 0)
        for i in range(num_base_layers):
            self.layers.insert(
                ((i + 1) * args.decoder_layers) // (num_base_layers + 1),
                BaseLayer(args),
            )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = TransformerDecoderLayer(args, no_encoder_attn)

        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint
            else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        **kwargs
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if not features_only:
            x = self.output_layer(x)

        # added
        # when decoding, will not be recorded after reordering encoder output
        if self.training or 'vqloss' in encoder_out:
            extra['vqloss'] = encoder_out['vqloss']

        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """

        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        # inner_states: List[Optional[Tensor]] = [x]

        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # added
            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                src_emb=encoder_out['encoder_embedding'][0],
                src_emb_mask=encoder_out['src_emb_padding_mask'][0],
            )

            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn] }

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[
                    embed_out_key
                ]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict
'''


def Embedding(num_embeddings, embedding_dim, padding_idx, init=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)

    # if init == 'pytorch':
    #     # N(0,1)
    #     pass
    # elif init == 'xavier':
    #     nn.init.xavier_uniform_(m.weight)
    # elif init == 'km':
    #     nn.init.kaiming_normal_(m.weight)
    # else:
    #     assert init is None
    #     nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)

    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


# @register_model_architecture("transformer", "transformer_tiny")
# def tiny_architecture(args):
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 64)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 64)
#     args.encoder_layers = getattr(args, "encoder_layers", 2)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
#     args.decoder_layers = getattr(args, "decoder_layers", 2)
#     args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 2)
#     return base_architecture(args)
#

@register_model_architecture("transformer_hybrid", "transformer_hybrid")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)


@register_model_architecture("transformer_hybrid", "transformer_hybrid_iwslt_de_en")
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)



# @register_model_architecture("transformer", "transformer_small")
# def transformer_iwslt_de_en(args):
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 256)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
#     args.encoder_layers = getattr(args, "encoder_layers", 4)
#     args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 256)
#     args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 256)
#     args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
#     args.decoder_layers = getattr(args, "decoder_layers", 4)
#     base_architecture(args)


@register_model_architecture("transformer_hybrid", "transformer_hybrid_wmt_en_de")
def transformer_wmt_en_de(args):
    base_architecture(args)
#
#
# # parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
# @register_model_architecture("transformer", "transformer_vaswani_wmt_en_de_big")
# def transformer_vaswani_wmt_en_de_big(args):
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
#     args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
#     args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
#     args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
#     args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
#     args.dropout = getattr(args, "dropout", 0.3)
#     base_architecture(args)
#
#
# @register_model_architecture("transformer", "transformer_vaswani_wmt_en_fr_big")
# def transformer_vaswani_wmt_en_fr_big(args):
#     args.dropout = getattr(args, "dropout", 0.1)
#     transformer_vaswani_wmt_en_de_big(args)
#
#
# @register_model_architecture("transformer", "transformer_wmt_en_de_big")
# def transformer_wmt_en_de_big(args):
#     args.attention_dropout = getattr(args, "attention_dropout", 0.1)
#     transformer_vaswani_wmt_en_de_big(args)
#
#
# # default parameters used in tensor2tensor implementation
# @register_model_architecture("transformer", "transformer_wmt_en_de_big_t2t")
# def transformer_wmt_en_de_big_t2t(args):
#     args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
#     args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
#     args.attention_dropout = getattr(args, "attention_dropout", 0.1)
#     args.activation_dropout = getattr(args, "activation_dropout", 0.1)
#     transformer_vaswani_wmt_en_de_big(args)


class MultiheadAttentionv2(MultiheadAttention):
    '''
    add 0/1 mask (float) dp attention
    add dp with edge feautures
    '''
    def __init__(
            self,
            embed_dim,
            num_heads,
            kdim=None,
            vdim=None,
            dropout=0.0,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            self_attention=False,
            encoder_decoder_attention=False,
            q_noise=0.0,
            qn_block_size=8,
    ):
        super().__init__(embed_dim, num_heads,
            kdim, vdim, dropout,bias,
            add_bias_kv, add_zero_attn,
            self_attention, encoder_decoder_attention,
            q_noise, qn_block_size,)

    def dot_with_features(self, x, y, z, transpose=True):
        pass

    def forward(
            self,
            query,
            key: Optional[Tensor],
            value: Optional[Tensor],
            attn_mask=None,
            edge_emb=None,
            key_padding_mask=None,
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            need_weights: bool = True,
            static_kv: bool = False,
            before_softmax: bool = False,
            need_head_weights: bool = False,
            **kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel

        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True

        is_tpu = query.device.type == "xla"

        # print('query.size(), key.size()', query.size(), key.size())

        tgt_len, bsz, embed_dim = query.size()

        # print(tgt_len, bsz)

        src_len = tgt_len
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if key is not None:
            src_len, key_bsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert key_bsz == bsz
                assert value is not None
                assert src_len, bsz == value.shape[:2]

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert self.encoder_decoder_attention and not self.self_attention
                    key = value = None
        else:
            saved_state = None

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        q *= self.scaling

        q = (
            q.contiguous()
                .view(tgt_len, bsz * self.num_heads, self.head_dim)
                .transpose(0, 1)
        )
        if k is not None:
            k = (
                k.contiguous()
                    .view(-1, bsz * self.num_heads, self.head_dim)
                    .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                    .view(-1, bsz * self.num_heads, self.head_dim)
                    .transpose(0, 1)
            )

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state["prev_key"] = k.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_value"] = v.view(bsz, self.num_heads, -1, self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state, saved_state)

        assert k is not None
        # added
        assert k.size(1) == src_len

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1
                )
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0), 1).type_as(
                            key_padding_mask
                        ),
                    ],
                    dim=1,
                )

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        if edge_emb is not None:
            # ''''''edge
            q_tmp = q.transpose(0, 1).view(bsz, self.num_heads, -1, self.head_dim)
            # q: bsz*t head h
            q_tmp = q_tmp.transpose(1, 2).reshape(-1, self.num_heads, self.head_dim)

            # edgeemd: b*t h t

            # B*T HEAD T
            attn_weights_edge = torch.bmm(q_tmp, edge_emb)
            attn_weights_edge = attn_weights_edge.view(bsz, tgt_len, self.num_heads, src_len)
            attn_weights_edge = attn_weights_edge.transpose(1, 2).reshape(-1, tgt_len, src_len)

            attn_weights = attn_weights + attn_weights_edge
            # print(attn_weights.size(), tgt_len, src_len)
            # ''''''edge

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        # added
        attn_weights = attn_weights - torch.max(attn_weights, -1, keepdim=True)[0]
        exp_logits = torch.exp(attn_weights)

        # B 1 1 S
        key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
        key_padding_mask = key_padding_mask.expand(-1, self.num_heads, -1, -1).reshape(-1, 1, src_len)

        exp_logits = exp_logits * key_padding_mask
        exp_sum_logits = torch.sum(exp_logits, -1, keepdims=True)
        attn_weights = exp_logits / (exp_sum_logits + 1e-8)

        # v2, which is better?
        # B N
        # attn_mask = (1 - attn_mask) * -1e8
        # attn_weights +=
        #

        # if attn_mask is not None:
        #     attn_mask = attn_mask.unsqueeze(0)
        #     if self.onnx_trace:
        #         attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
        #
        #     attn_weights += attn_mask
        #
        # if key_padding_mask is not None:
        #     # don't attend to padding symbols
        #     attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        #     # added
        #
        #     if not is_tpu:
        #         attn_weights = attn_weights.masked_fill(
        #             key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
        #             float("-inf"),
        #         )
        #
        #     attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights, v

        attn_weights_float = utils.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace
        )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = self.dropout_module(attn_weights)

        assert v is not None
        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        if edge_emb is not None:
            # ''''''edge
            # bsz * head t, t
            attn_probs_tmp = attn_probs.view(bsz, self.num_heads, -1, tgt_len)
            # q: bsz*t head t
            attn_probs_tmp = attn_probs_tmp.transpose(1, 2).reshape(-1, self.num_heads, tgt_len)

            # edgeemd: b*t h t

            # q: bsz*t head h
            attn_edge = torch.bmm(attn_probs_tmp, edge_emb.transpose(1, 2))
            attn_edge = attn_edge.view(bsz, tgt_len, self.num_heads, -1)
            attn_edge = attn_edge.transpose(1, 2).reshape(-1, tgt_len, self.head_dim)

            attn = attn + attn_edge
            # '''''

        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None

        if need_weights:
            attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).transpose(1, 0)

            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)

        return attn, attn_weights


class TransformerGraphLayer(TransformerEncoderLayer):

    def __init__(self, args):
        super().__init__(args, )
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = self.build_self_attention(self.embed_dim, args)

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttentionv2(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=False, # added
            # kdim=embed_dim, vdim=embed_dim*2, # added
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def forward(self, x, mask, edge_emb):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        # if attn_mask is not None:
        #     attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=mask,
            need_weights=False,
            edge_emb=edge_emb,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x

