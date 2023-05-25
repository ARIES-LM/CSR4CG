# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention, MultiheadAttentionWithRelPos
from fairseq.modules import MultiheadAttentionWithRelPosCrossBoundry
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.fairseq_dropout import FairseqDropout

# added
from fairseq.modules.vector_quantize import VectorQuantize

from torch import Tensor

# import torch.nn.functional as F
import torch.distributed as dist

class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.use_rel_pos = getattr(args, "use_rel_pos", False)
        if self.use_rel_pos:
            self.self_attn = self.build_self_attention_rel_pos(self.embed_dim, args)
        else:
            self.self_attn = self.build_self_attention(self.embed_dim, args)

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim, args.encoder_ffn_embed_dim, self.quant_noise, self.quant_noise_block_size
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim, self.embed_dim, self.quant_noise, self.quant_noise_block_size
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_self_attention_rel_pos(self, embed_dim, args):
        return MultiheadAttentionWithRelPos(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            max_position_embeddings = args.max_relative_position,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None, **kwargs):
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
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x

        if not self.normalize_before:
            x = self.final_layer_norm(x)

        return x



class TransformerEncoderLayerVQ(TransformerEncoderLayer):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__(args)
        self.embed_dim = args.encoder_embed_dim
        embed_dim = self.embed_dim

        # added
        self.vqmode = args.vqmode
        # first get cookbook using pretrained models
        # then only train a part of models
        vqdim = args.vqdim
        k = args.k[0]

        self.minibz = args.kmbatch_multiplier * max(args.k)

        if vqdim != embed_dim:
            self.tovq = Linear(embed_dim, vqdim)
            self.toemb = Linear(vqdim, embed_dim)
            # self.toemb = Linear(vqdim+embed_dim, embed_dim)
        else:
            self.toemb = Linear(embed_dim, embed_dim)

        if len(args.k) > 1:
            self.vqatt = self.build_attention(embed_dim, vqdim, args)

        if not args.vq_share:
            self.vq = VectorQuantize(
                dim=vqdim,
                n_embed=k,
                decay=args.vq_decay, # ema update
                commitment=args.vq_commitment, # loss weight
                wait_steps=args.vq_wait_steps,
                observe_steps=args.vq_observe_steps, # kmeans initialization
                coreset_size_multiplier=args.kmbatch_multiplier,
                usecosine=args.vq_cosine,
            )
        self.vqshare = args.vq_share
        self.vqdp = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.vq_layer_norm = LayerNorm(self.embed_dim)

        self.vqdrop = args.vq_drop

        # histogram
        self.print_code_usage = False
        self.availability = {i:0 for i in range(k)}

    def build_attention(self, qdim, kvdim, args):
        return MultiheadAttention(
            qdim,
            args.encoder_attention_heads,
            kdim=kvdim,
            vdim=kvdim,
            dropout=args.attention_dropout,
            self_attention=False,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None,
                quantization_mask=None,for_warmup=False, vqmodel=None):
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

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # added
        categories = None
        if self.vqmode:
            if not self.vqshare:
                # to B T H
                if hasattr(self, 'tovq'):
                    projected_features = self.tovq(x).transpose(0, 1)
                else:
                    projected_features = x.transpose(0, 1)

                unquantized_features = projected_features[quantization_mask]

                # mean loss
                (quantized_features, categories, commit_loss, dist) = self.vq(unquantized_features, self.minibz)

                if dist is not None:
                    # for analysis
                    # src_cats = x.new_full((x.size(1), x.size(0)), -1).long()
                    # src_cats[quantization_mask] = categories
                    # for c in src_cats:
                    #     print(c)
                    #
                    # exit()

                    isspecial = ~quantization_mask

                    residual = x

                    extra_content_annotations = torch.zeros_like(projected_features)
                    extra_content_annotations[quantization_mask] = quantized_features

                    # T B H
                    extra_content_annotations = extra_content_annotations.transpose(0, 1)

                    # expand hidden size
                    extra_content_annotations = self.toemb(extra_content_annotations)
                    # extra_content_annotations = self.vqdp(extra_content_annotations)

                    # if self.training and self.vqdrop > 0:
                    #     # T B
                    #     probs = residual.new_ones(extra_content_annotations.size(0),
                    #                               extra_content_annotations.size(1)).uniform_()
                    #
                    #     dropresidual = (probs < self.vqdrop).unsqueeze(-1)
                    #     x = residual.masked_fill(dropresidual, 0) + extra_content_annotations

                    if not self.normalize_before:
                        x = self.vq_layer_norm(x)

                    x[isspecial.T] = residual[isspecial.T]
            else:
                commit_loss = 0
                if for_warmup:
                    if vqmodel is not None:
                        # use features of a specific layer for kmeans warmup

                        # to B T H
                        if hasattr(self, 'tovq'):
                            projected_features = self.tovq(x).transpose(0, 1)
                        else:
                            projected_features = x.transpose(0, 1)

                        unquantized_features = projected_features[quantization_mask]

                        if type(vqmodel) == nn.ModuleList:
                            for onevq in vqmodel:
                                (quantized_features, categories, commit_loss, dist) = onevq(unquantized_features, self.minibz)
                        else:
                            (quantized_features, categories, commit_loss, dist) = vqmodel(unquantized_features, self.minibz)
                        # print(commit_loss)
                        # assert commit_loss.item() == 0
                else:
                    # to B T H
                    if hasattr(self, 'tovq'):
                        projected_features = self.tovq(x).transpose(0, 1)
                    else:
                        projected_features = x.transpose(0, 1)

                    unquantized_features = projected_features[quantization_mask]

                    # mean loss
                    if type(vqmodel) == nn.ModuleList:
                        exit('do not use multi vq')

                        multi_scale_features = []
                        for onevq in vqmodel:
                            (quantized_feature_vq, categories, commit_loss_vq, dist) = onevq(unquantized_features,
                                                                                          self.minibz)
                            commit_loss += commit_loss_vq

                            extra_content_annotations = torch.zeros_like(projected_features)
                            extra_content_annotations[quantization_mask] = quantized_feature_vq

                            multi_scale_features.append(extra_content_annotations)

                        commit_loss = commit_loss / len(vqmodel)
                        # M B T H
                        multi_scale_features = torch.stack(multi_scale_features, 0)

                        # attention
                        scale, bisze, tsize, hsize = multi_scale_features.size()
                        # M TB H
                        multi_scale_features = multi_scale_features.transpose(1, 2).reshape(scale, -1, hsize)

                        residual = x

                        if self.normalize_before:
                            x = self.vq_layer_norm(x)

                        # 1 TB H
                        x = x.view(1, -1, hsize)

                        x, _ = self.vqatt(
                            query=x,
                            key=multi_scale_features,
                            value=multi_scale_features,
                        )
                        x = self.dropout_module(x)

                        x = x.view(tsize, bisze, -1)

                        x = residual + x

                        if not self.normalize_before:
                            x = self.vq_layer_norm(x)

                    else:
                        (quantized_features, categories, commit_loss, dist) = vqmodel(unquantized_features, self.minibz)

                        # for analysis
                        # src_cats = x.new_full((x.size(1), x.size(0)), -1).long()
                        # src_cats[quantization_mask] = categories
                        # for c in src_cats:
                        #     print(c)
                        # exit()

                        isspecial = ~quantization_mask

                        residual = x

                        extra_content_annotations = torch.zeros_like(projected_features)
                        extra_content_annotations[quantization_mask] = quantized_features

                        # expand hidden size
                        extra_content_annotations = self.toemb(extra_content_annotations)
                        extra_content_annotations = self.vqdp(extra_content_annotations)
                        # T B H
                        extra_content_annotations = extra_content_annotations.transpose(0, 1)

                        x = residual + extra_content_annotations

                        if not self.normalize_before:
                            x = self.vq_layer_norm(x)

                        x[isspecial.T] = residual[isspecial.T]
        else:
            commit_loss = 0

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x

        if not self.normalize_before:
            x = self.final_layer_norm(x)

        # for analysis
        if self.print_code_usage:
            for ki in self.availability:
                code_i_use = (categories == ki).sum().item()
                self.availability[ki] += code_i_use

        return {'x': x, 'loss': commit_loss, 'codes': categories}



class TransformerEncoderLayerVQv2(TransformerEncoderLayerVQ):
    def __init__(self, args):
        super().__init__(args)

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None,
                quantization_mask=None,for_warmup=False, vqmodel=None):
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x

        if not self.normalize_before:
            x = self.final_layer_norm(x)

        # added
        categories = None
        if self.vqmode:
            if not self.vqshare:
                # to B T H
                if hasattr(self, 'tovq'):
                    projected_features = self.tovq(x).transpose(0, 1)
                else:
                    projected_features = x.transpose(0, 1)

                unquantized_features = projected_features[quantization_mask]

                # mean loss
                (quantized_features, categories, commit_loss, dist) = self.vq(unquantized_features, self.minibz)

                if dist is not None:
                    # for analysis
                    # src_cats = x.new_full((x.size(1), x.size(0)), -1).long()
                    # src_cats[quantization_mask] = categories
                    # for c in src_cats:
                    #     print(c)
                    #
                    # exit()

                    isspecial = ~quantization_mask

                    residual = x

                    extra_content_annotations = torch.zeros_like(projected_features)
                    extra_content_annotations[quantization_mask] = quantized_features

                    # T B H
                    extra_content_annotations = extra_content_annotations.transpose(0, 1)

                    # expand hidden size
                    extra_content_annotations = self.toemb(extra_content_annotations)
                    extra_content_annotations = self.vqdp(extra_content_annotations)
                    x = residual + extra_content_annotations

                    if not self.normalize_before:
                        x = self.vq_layer_norm(x)

                    x[isspecial.T] = residual[isspecial.T]
            else:
                commit_loss = 0
                if for_warmup:
                    if vqmodel is not None:
                        # use features of a specific layer for kmeans warmup

                        # to B T H
                        if hasattr(self, 'tovq'):
                            projected_features = self.tovq(x).transpose(0, 1)
                        else:
                            projected_features = x.transpose(0, 1)

                        unquantized_features = projected_features[quantization_mask]

                        if type(vqmodel) == nn.ModuleList:
                            for onevq in vqmodel:
                                (quantized_features, categories, commit_loss, dist) = onevq(unquantized_features, self.minibz)
                        else:
                            (quantized_features, categories, commit_loss, dist) = vqmodel(unquantized_features, self.minibz)
                        # print(commit_loss)
                        # assert commit_loss.item() == 0
                else:
                    # to B T H
                    if hasattr(self, 'tovq'):
                        projected_features = self.tovq(x).transpose(0, 1)
                    else:
                        projected_features = x.transpose(0, 1)

                    unquantized_features = projected_features[quantization_mask]

                    # mean loss
                    if type(vqmodel) == nn.ModuleList:
                        exit('do not use multi vq')

                        multi_scale_features = []
                        for onevq in vqmodel:
                            (quantized_feature_vq, categories, commit_loss_vq, dist) = onevq(unquantized_features,
                                                                                          self.minibz)
                            commit_loss += commit_loss_vq

                            extra_content_annotations = torch.zeros_like(projected_features)
                            extra_content_annotations[quantization_mask] = quantized_feature_vq

                            multi_scale_features.append(extra_content_annotations)

                        commit_loss = commit_loss / len(vqmodel)
                        # M B T H
                        multi_scale_features = torch.stack(multi_scale_features, 0)

                        # attention
                        scale, bisze, tsize, hsize = multi_scale_features.size()
                        # M TB H
                        multi_scale_features = multi_scale_features.transpose(1, 2).reshape(scale, -1, hsize)

                        residual = x

                        if self.normalize_before:
                            x = self.vq_layer_norm(x)

                        # 1 TB H
                        x = x.view(1, -1, hsize)

                        x, _ = self.vqatt(
                            query=x,
                            key=multi_scale_features,
                            value=multi_scale_features,
                        )
                        x = self.dropout_module(x)

                        x = x.view(tsize, bisze, -1)

                        x = residual + x

                        if not self.normalize_before:
                            x = self.vq_layer_norm(x)

                    else:
                        (quantized_features, categories, commit_loss, dist) = vqmodel(unquantized_features, self.minibz)

                        # for analysis
                        # src_cats = x.new_full((x.size(1), x.size(0)), -1).long()
                        # src_cats[quantization_mask] = categories
                        # for c in src_cats:
                        #     print(c)
                        # exit()

                        isspecial = ~quantization_mask

                        residual = x

                        extra_content_annotations = torch.zeros_like(projected_features)
                        extra_content_annotations[quantization_mask] = quantized_features

                        # expand hidden size
                        extra_content_annotations = self.toemb(extra_content_annotations)
                        extra_content_annotations = self.vqdp(extra_content_annotations)
                        # T B H
                        extra_content_annotations = extra_content_annotations.transpose(0, 1)

                        x = residual + extra_content_annotations

                        if not self.normalize_before:
                            x = self.vq_layer_norm(x)

                        x[isspecial.T] = residual[isspecial.T]
        else:
            commit_loss = 0

        # for analysis
        if self.print_code_usage:
            for ki in self.availability:
                code_i_use = (categories == ki).sum().item()
                self.availability[ki] += code_i_use

        return {'x': x, 'loss': commit_loss, 'codes': categories}



class TransformerEncoderLayerVQGate(TransformerEncoderLayer):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__(args)
        self.embed_dim = args.encoder_embed_dim
        embed_dim = self.embed_dim

        # added
        self.vqmode = args.vqmode
        # first get cookbook using pretrained models
        # then only train a part of models
        vqdim = args.vqdim
        k = args.k[0]

        self.minibz = args.kmbatch_multiplier * max(args.k)

        if vqdim != embed_dim:
            self.tovq = Linear(embed_dim, vqdim, False)
            self.toemb = Linear(vqdim, embed_dim, False)
            # self.toemb = Linear(vqdim+embed_dim, embed_dim)
        else:
            self.toemb = Linear(embed_dim, embed_dim)

        self.gate = Linear(embed_dim*2, 1, False)

        if len(args.k) > 1:
            self.vqatt = self.build_attention(embed_dim, vqdim, args)

        if not args.vq_share:
            self.vq = VectorQuantize(
                dim=vqdim,
                n_embed=k,
                decay=args.vq_decay, # ema update
                commitment=args.vq_commitment, # loss weight
                wait_steps=args.vq_wait_steps,
                observe_steps=args.vq_observe_steps, # kmeans initialization
                coreset_size_multiplier=args.kmbatch_multiplier,
                usecosine=args.vq_cosine,
            )
        self.vqshare = args.vq_share
        # self.vqdp = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        # self.vq_layer_norm = LayerNorm(self.embed_dim)

        # histogram
        self.availability = {i:0 for i in range(k)}
        self.print_code_usage = False

    def build_attention(self, qdim, kvdim, args):
        return MultiheadAttention(
            qdim,
            args.encoder_attention_heads,
            kdim=kvdim,
            vdim=kvdim,
            dropout=args.attention_dropout,
            self_attention=False,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None,
                quantization_mask=None,for_warmup=False, vqmodel=None):
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

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x

        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )

        # added
        if self.vqmode:
            if not self.vqshare:
                # to B T H
                if hasattr(self, 'tovq'):
                    projected_features = self.tovq(x).transpose(0, 1)
                else:
                    projected_features = x.transpose(0, 1)

                unquantized_features = projected_features[quantization_mask]

                # mean loss
                (quantized_features, categories, commit_loss, dist) = self.vq(unquantized_features, self.minibz)

                if dist is not None:
                    # for analysis
                    # src_cats = x.new_full((x.size(1), x.size(0)), -1).long()
                    # src_cats[quantization_mask] = categories
                    # for c in src_cats:
                    #     print(c)
                    #
                    # exit()

                    isspecial = ~quantization_mask

                    attoutput = x

                    extra_content_annotations = torch.zeros_like(projected_features)
                    extra_content_annotations[quantization_mask] = quantized_features

                    # T B H
                    extra_content_annotations = extra_content_annotations.transpose(0, 1)

                    # expand hidden size
                    x = self.toemb(extra_content_annotations)

                    g = torch.sigmoid(self.gate(torch.cat((attoutput, x), -1)))

                    x = x * g + (1-g) * attoutput

                    x[isspecial.T] = attoutput[isspecial.T]
            else:
                commit_loss = 0
                if for_warmup:
                    if vqmodel is not None:
                        # use features of a specific layer for kmeans warmup

                        # to B T H
                        if hasattr(self, 'tovq'):
                            projected_features = self.tovq(x).transpose(0, 1)
                        else:
                            projected_features = x.transpose(0, 1)

                        unquantized_features = projected_features[quantization_mask]

                        if type(vqmodel) == nn.ModuleList:
                            for onevq in vqmodel:
                                (quantized_features, categories, commit_loss, dist) = onevq(unquantized_features, self.minibz)
                        else:
                            (quantized_features, categories, commit_loss, dist) = vqmodel(unquantized_features, self.minibz)
                        # print(commit_loss)
                        # assert commit_loss.item() == 0
                else:
                    # to B T H
                    if hasattr(self, 'tovq'):
                        projected_features = self.tovq(x).transpose(0, 1)
                    else:
                        projected_features = x.transpose(0, 1)

                    unquantized_features = projected_features[quantization_mask]

                    # mean loss
                    if type(vqmodel) == nn.ModuleList:
                        exit('do not use multi vq')

                        multi_scale_features = []
                        for onevq in vqmodel:
                            (quantized_feature_vq, categories, commit_loss_vq, dist) = onevq(unquantized_features,
                                                                                          self.minibz)
                            commit_loss += commit_loss_vq

                            extra_content_annotations = torch.zeros_like(projected_features)
                            extra_content_annotations[quantization_mask] = quantized_feature_vq

                            multi_scale_features.append(extra_content_annotations)

                        commit_loss = commit_loss / len(vqmodel)
                        # M B T H
                        multi_scale_features = torch.stack(multi_scale_features, 0)

                        # attention
                        scale, bisze, tsize, hsize = multi_scale_features.size()
                        # M TB H
                        multi_scale_features = multi_scale_features.transpose(1, 2).reshape(scale, -1, hsize)

                        residual = x

                        if self.normalize_before:
                            x = self.vq_layer_norm(x)

                        # 1 TB H
                        x = x.view(1, -1, hsize)

                        x, _ = self.vqatt(
                            query=x,
                            key=multi_scale_features,
                            value=multi_scale_features,
                        )
                        x = self.dropout_module(x)

                        x = x.view(tsize, bisze, -1)

                        x = residual + x

                        if not self.normalize_before:
                            x = self.vq_layer_norm(x)

                    else:
                        (quantized_features, categories, commit_loss, dist) = vqmodel(unquantized_features, self.minibz)

                        # for analysis
                        # src_cats = x.new_full((x.size(1), x.size(0)), -1).long()
                        # src_cats[quantization_mask] = categories
                        # for c in src_cats:
                        #     print(c)
                        # exit()

                        isspecial = ~quantization_mask

                        residual = x

                        extra_content_annotations = torch.zeros_like(projected_features)
                        extra_content_annotations[quantization_mask] = quantized_features

                        # expand hidden size
                        extra_content_annotations = self.toemb(extra_content_annotations)
                        extra_content_annotations = self.vqdp(extra_content_annotations)
                        # T B H
                        extra_content_annotations = extra_content_annotations.transpose(0, 1)

                        x = residual + extra_content_annotations

                        if not self.normalize_before:
                            x = self.vq_layer_norm(x)

                        x[isspecial.T] = residual[isspecial.T]
        else:
            commit_loss = 0


        x = self.dropout_module(x)
        x = residual + x

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)


        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x

        if not self.normalize_before:
            x = self.final_layer_norm(x)

        # for analysis
        if self.print_code_usage:
            for ki in self.availability:
                code_i_use = (categories == ki).sum().item()
                self.availability[ki] += code_i_use

        return {'x': x, 'loss': commit_loss}


class TransformerSetEncoderLayerv2(TransformerEncoderLayer):
    def __init__(self, args):
        super().__init__(args)
        # export = getattr(args, "export", False)

        self.pack_attn = self.build_pack_attention(self.embed_dim, args)
        self.pack_attn_layer_norm = LayerNorm(self.embed_dim)

        # self.pack_self_attn_layer_norm = LayerNorm(self.embed_dim)
        # self.pack_self_attn = self.build_pack_attention(self.embed_dim, args)

        self.mpffn = args.mpffn
        if self.mpffn:
            self.mpfc1 = self.build_fc1(
                self.embed_dim,
                args.encoder_ffn_embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.mpfc2 = self.build_fc2(
                args.encoder_ffn_embed_dim,
                self.embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.mpffn_layer_norm = LayerNorm(self.embed_dim)

        self.unpack_layer_norm = LayerNorm(self.embed_dim)
        self.unpack_attn = self.build_pack_attention(self.embed_dim, args)

        self.pack_num = args.pack_num
        self.batchmp = False
        self.fullatt = args.fullatt

        self.topk = args.qtopk
        self.ablation = args.ablation

        self.distance = args.distance
        self.mixlamb = args.mixlamb
        self.temp = args.mixtemp

    # added
    def build_pack_attention(self, embed_dim, args):
        # just common attention
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=False,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    # added
    def build_pack_attentionv2(self, qdim, kdim, vdim, args):
        # just common attention
        return MultiheadAttention(
            qdim,
            args.encoder_attention_heads,
            kdim=kdim,
            vdim=vdim,
            dropout=args.attention_dropout,
            self_attention=False,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def residual_connection(self, x, y):
        return x + y

    def forward(
            self,
            x,
            encoder_padding_mask: Optional[Tensor],
            attn_mask: Optional[Tensor] = None,
            packed_x=None,
    ):

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # added
        batchmp = self.batchmp

        if self.training and batchmp:
            psize, bsize, _ = packed_x.size()

            # pack
            residual_px = packed_x
            if self.normalize_before:
                packed_x = self.pack_attn_layer_norm(packed_x)

            # P B H
            packed_x, _ = self.pack_attn(
                query=packed_x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
            )
            packed_x = self.dropout_module(packed_x)
            packed_x = self.residual_connection(packed_x, residual_px)
            if not self.normalize_before:
                packed_x = self.pack_attn_layer_norm(packed_x)

            # exchange
            if not self.ablation:

                # if self.topk > 0:
                #     if self.topk >= packed_x.size(1):
                #         selected = packed_x.unsqueeze(1).expand(-1, bsize, -1, -1)
                #     else:
                #         with torch.no_grad():
                #             # P B B
                #             # eu or cosine?
                #             distance = torch.cdist(packed_x, packed_x)
                #
                #             # non-para mix
                #             # remove self
                #             eye = torch.eye(bsize, dtype=encoder_padding_mask.dype, devive=packed_x.device)
                #             eye = eye.unsqueeze(0).expand_as(distance)
                #             distance = distance.masked_fill(eye, float('-inf'))
                #
                #             weight_float = utils.softmax(distance / self.temp, -1)
                #             weight = weight_float.type_as(distance)
                #             # todo: dropout weight?
                #             neib_packed_x = torch.bmm(weight, packed_x)
                #             # todo: add randomness for mixlamb
                #             mixed_packed_x = self.mixlamb * packed_x + (1 - self.mixlamb) * neib_packed_x
                #
                #             # P B k
                #             _, smallk_index = torch.topk(distance, self.topk, dim=2, largest=False)
                #
                #         # P B * H
                #         total_packed_x = packed_x.unsqueeze(1).expand(-1, bsize, -1, -1)
                #         selected = torch.gather(total_packed_x, dim=2,
                #                                 index=smallk_index.unsqueeze(-1).expand(-1, -1, -1, packed_x.size(-1)))

                # residual_px = packed_x
                # if self.normalize_before:
                #     packed_x = self.pack_self_attn_layer_norm(packed_x)
                #
                # if self.fullatt:
                #     # PB 1 H
                #     packed_x = packed_x.reshape(-1, 1, packed_x.size(-1))
                # else:
                #     # B P H
                #     packed_x = packed_x.transpose(0, 1)

                # for multi-gpu
                # if torch.cuda.device_count() > 1 and dist.get_world_size() > 1:
                #     cuda_device = f'cuda:{dist.get_rank()}'
                #     local_size = torch.tensor(packed_x.size()[0], device=cuda_device)
                #
                #     ws = dist.get_world_size()
                #     all_sizes = [torch.zeros_like(local_size) for _ in range(ws)]
                #     dist.all_gather(all_sizes, local_size)
                #
                #     # print(all_sizes)
                #     max_size = max(all_sizes)
                #     # print(max_size)
                #
                #     size_diff = max_size.item() - local_size.item()
                #     if size_diff:
                #         padding = packed_x.new_zeros(size_diff, psize, packed_x.size(-1))
                #         # print(padding.size(), packed_x.size())
                #         pad_packed_x = torch.cat((packed_x, padding), 0)
                #     else:
                #         pad_packed_x = packed_x
                #
                #     all_qs_padded = [torch.zeros_like(pad_packed_x) for _ in range(ws)]
                #     dist.all_gather(all_qs_padded, pad_packed_x)
                #     all_qs = []
                #     for q, size in zip(all_qs_padded, all_sizes):
                #         all_qs.append(q[:size])
                #
                #     # for gradient
                #     all_qs[dist.get_rank()] = packed_x
                #     packed_x_gather = torch.cat(all_qs, 0)
                #     # print(packed_x_gather.size())
                #     # exit()
                #
                #     # packed_x_list = [torch.zeros_like(packed_x) for _ in range(dist.get_world_size())]
                #     # dist.all_gather(packed_x_list, tensor=packed_x)
                #     # packed_x_list[dist.get_rank()] = packed_x
                #     # # B * GPU P H
                #     # packed_x_gather = torch.cat(packed_x_list, 0)
                # else:

                # for topk
                # 1 PB H
                # packed_x = packed_x.reshape(-1, 1, packed_x.size(-1)).transpose(0, 1)
                # # k PB H
                # selected = selected.reshape(-1, selected.size(2), selected.size(3)).transpose(0, 1)
                # # 1 PB H
                # packed_x, _ = self.pack_self_attn(
                #     query=packed_x,
                #     key=selected,
                #     value=selected,
                # )
                # packed_x = packed_x.reshape(psize, bsize, -1)


                # packed_x_gather = packed_x
                # packed_x, _ = self.pack_self_attn(
                #     query=packed_x,
                #     key=packed_x_gather,
                #     value=packed_x_gather,
                # )
                #
                # # P B H
                # if self.fullatt:
                #     packed_x = packed_x.reshape(psize, bsize, -1)
                # else:
                #     packed_x = packed_x.transpose(0, 1)
                #
                # packed_x = self.dropout_module(packed_x)
                # packed_x = self.residual_connection(packed_x, residual_px)
                # if not self.normalize_before:
                #     packed_x = self.pack_self_attn_layer_norm(packed_x)


                # non-para

                with torch.no_grad():
                    # P B B
                    # eu or cosine?
                    if self.distance == 'cos':
                        packed_x_norm = torch.nn.functional.normalize(packed_x, dim=-1)
                    else:
                        packed_x_norm = packed_x

                    distance = torch.cdist(packed_x_norm, packed_x_norm)

                    # non-para mix
                    # remove self
                    eye = torch.eye(bsize).cuda().bool()
                    eye = eye.unsqueeze(0).expand_as(distance)
                    distance.masked_fill_(eye, float('-inf'))

                    weight_float = utils.softmax(distance / self.temp, -1)
                    weight = weight_float.type_as(distance)

                    # todo: dropout weight
                    # P B H
                    neib_packed_x = torch.bmm(weight, packed_x)

                packed_x = self.mixlamb * packed_x + (1 - self.mixlamb) * neib_packed_x

                if self.mpffn:
                    residual_px = packed_x
                    if self.normalize_before:
                        packed_x = self.mpffn_layer_norm(packed_x)
                    packed_x = self.activation_fn(self.mpfc1(packed_x))
                    packed_x = self.activation_dropout_module(packed_x)
                    packed_x = self.mpfc2(packed_x)
                    packed_x = self.dropout_module(packed_x)
                    packed_x = self.residual_connection(packed_x, residual_px)
                    if not self.normalize_before:
                        packed_x = self.mpffn_layer_norm(packed_x)

            # unpack
            residual = x
            if self.normalize_before:
                x = self.unpack_layer_norm(x)

            x, _ = self.unpack_attn(
                query=x,
                key=packed_x,
                value=packed_x,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.unpack_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        # added
        if self.training and self.batchmp:
            output = {'x': x, 'packed_x': packed_x}
            return output

        return x



class TransformerEncoderLayerMFFN(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.use_rel_pos = getattr(args, "use_rel_pos", False)
        if self.use_rel_pos:
            self.self_attn = self.build_self_attention_rel_pos(self.embed_dim, args)
        else:
            self.self_attn = self.build_self_attention(self.embed_dim, args)

        # added
        self.headnum = args.encoder_attention_heads
        self.rezero = args.rezero

        if self.rezero:
            self.self_attn_layer_norm = None
            self.register_parameter('rezero_w1', nn.Parameter(torch.Tensor([0])))
        else:
            self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before

        ffnkvnums = args.ffnkvnum
        self.ffn1 = nn.ModuleList()
        self.ffn2 = nn.ModuleList()
        for middim in ffnkvnums:
            fc1 = self.build_fc1(
                self.embed_dim // self.headnum, middim, self.quant_noise, self.quant_noise_block_size
            )
            fc2 = self.build_fc2(
                middim, self.embed_dim//self.headnum, self.quant_noise, self.quant_noise_block_size
            )
            self.ffn1.append(fc1)
            self.ffn2.append(fc2)

        # added
        if self.rezero:
            self.final_layer_norm = None
            self.register_parameter('rezero_w2', nn.Parameter(torch.Tensor([0])))
        else:
            self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_self_attention_rel_pos(self, embed_dim, args):
        return MultiheadAttentionWithRelPos(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            max_position_embeddings = args.max_relative_position,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, attn_mask: Optional[Tensor] = None):
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
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x

        if self.normalize_before:
            # added
            if self.self_attn_layer_norm is not None:
                x = self.self_attn_layer_norm(x)

        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)

        # added
        if self.rezero:
            x = residual + x * self.rezero_w1
        else:
            x = residual + x

        if not self.normalize_before:
            # added
            if self.self_attn_layer_norm is not None:
                x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            # added
            if self.final_layer_norm is not None:
                x = self.final_layer_norm(x)

        x_splits = x.split(self.embed_dim//self.headnum, dim=-1)
        xs_ffn = []
        for i, x_head in enumerate(x_splits):
            x_head = self.activation_fn(self.ffn1[i](x_head))
            x_head = self.activation_dropout_module(x_head)
            x_head = self.ffn2[i](x_head)
            x_head = self.dropout_module(x_head)
            xs_ffn.append(x_head)
        x = torch.cat(xs_ffn, -1)

        # x = self.activation_fn(self.fc1(x))
        # x = self.activation_dropout_module(x)
        # x = self.fc2(x)
        # x = self.dropout_module(x)

        # added
        if self.rezero:
            x = residual + x * self.rezero_w1
        else:
            x = residual + x

        if not self.normalize_before:
            # added
            if self.final_layer_norm is not None:
                x = self.final_layer_norm(x)

        return x



# encode both source and target together
class TransformerEncoderLayerCrossBoundary(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.use_rel_pos = getattr(args, "use_rel_pos", False)
        if self.use_rel_pos:
            self.self_attn = self.build_self_attention_rel_pos(self.embed_dim, args)
        else:
            self.self_attn = self.build_self_attention(self.embed_dim, args)

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim, args.encoder_ffn_embed_dim, self.quant_noise, self.quant_noise_block_size
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim, self.embed_dim, self.quant_noise, self.quant_noise_block_size
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_self_attention(self, embed_dim, args):
        encoder_attention_heads = args.dangle_encoder_attention_heads if hasattr(args, "dangle_encoder_attention_heads") else args.encoder_attention_heads
        return MultiheadAttention(
            embed_dim,
            encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_self_attention_rel_pos(self, embed_dim, args):
        encoder_attention_heads = args.dangle_encoder_attention_heads if hasattr(args, "dangle_encoder_attention_heads") else args.encoder_attention_heads
        return MultiheadAttentionWithRelPosCrossBoundry(
            embed_dim,
            encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            max_position_embeddings = args.max_relative_position,
        )


    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, part1_len, part2_len, attn_mask: Optional[Tensor] = None, need_weights=False):
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
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # print("relations : "+str(relation))
        
        if self.use_rel_pos:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
                need_weights=need_weights,
                part1_len=part1_len,
                part2_len=part2_len,
            )
        else:
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                attn_mask=attn_mask,
                need_weights=need_weights
            )
            
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if attn is None:
            return x
        else:
            return x, attn


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
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.use_rel_pos = getattr(args, "use_rel_pos", False)
        if self.use_rel_pos:
            self.self_attn = self.build_self_attention_rel_pos(self.embed_dim, args, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn)
        else:
            self.self_attn = self.build_self_attention(self.embed_dim, args, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn)

        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__)
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)

        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim, args.decoder_ffn_embed_dim, self.quant_noise, self.quant_noise_block_size
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim, self.embed_dim, self.quant_noise, self.quant_noise_block_size
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
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


    def build_self_attention_rel_pos(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        return MultiheadAttentionWithRelPos(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            max_position_embeddings = args.max_relative_position,
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
            if self.self_attn_layer_norm is not None:
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
        x = residual + x

        if not self.normalize_before:
            if self.self_attn_layer_norm is not None:
                x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None:
            residual = x

            if self.normalize_before:
                if self.encoder_attn_layer_norm is not None:
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

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = residual + x

            if not self.normalize_before:
                if self.encoder_attn_layer_norm is not None:
                    x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            if self.final_layer_norm is not None:
                x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x

        if not self.normalize_before:
            if self.final_layer_norm is not None:
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


class TransformerDecoderLayerVQ(TransformerDecoderLayer):
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False
    ):
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)

        self.embed_dim = args.encoder_embed_dim
        embed_dim = self.embed_dim

        # added
        self.vqmode = args.vqmode
        # first get cookbook using pretrained models
        # then only train a part of models
        vqdim = args.vqdim
        k = args.k[0]

        self.minibz = args.kmbatch_multiplier * max(args.k)

        if vqdim != embed_dim:
            self.tovq = Linear(embed_dim, vqdim)
            self.toemb = Linear(vqdim, embed_dim)
        else:
            self.toemb = Linear(embed_dim, embed_dim)

        if len(args.k) > 1:
            self.vqatt = self.build_attention(embed_dim, vqdim, args)

        if not args.vq_share:
            self.vq = VectorQuantize(
                dim=vqdim,
                n_embed=k,
                decay=args.vq_decay, # ema update
                commitment=args.vq_commitment, # loss weight
                wait_steps=args.vq_wait_steps,
                observe_steps=args.vq_observe_steps, # kmeans initialization
                coreset_size_multiplier=args.kmbatch_multiplier,
                usecosine=args.vq_cosine,
            )
        self.vqshare = args.vq_share
        self.vqdp = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.vq_layer_norm = LayerNorm(self.embed_dim)

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
        quantization_mask=None, for_warmup=False, vqmodel=None
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
        x = residual + x

        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)


        # added
        if self.vqmode:
            if not self.vqshare:
                # to B T H
                if hasattr(self, 'tovq'):
                    projected_features = self.tovq(x).transpose(0, 1)
                else:
                    projected_features = x.transpose(0, 1)

                unquantized_features = projected_features[quantization_mask]

                # mean loss
                (quantized_features, categories, commit_loss, dist) = self.vq(unquantized_features, self.minibz)

                if dist is not None:
                    # for analysis
                    # src_cats = x.new_full((x.size(1), x.size(0)), -1).long()
                    # src_cats[quantization_mask] = categories
                    # for c in src_cats:
                    #     print(c)
                    isspecial = ~quantization_mask

                    residual = x

                    extra_content_annotations = torch.zeros_like(projected_features)
                    extra_content_annotations[quantization_mask] = quantized_features

                    # expand hidden size
                    extra_content_annotations = self.toemb(extra_content_annotations)
                    extra_content_annotations = self.vqdp(extra_content_annotations)
                    # T B H
                    extra_content_annotations = extra_content_annotations.transpose(0, 1)

                    x = residual + extra_content_annotations

                    if not self.normalize_before:
                        x = self.vq_layer_norm(x)

                    x[isspecial.T] = residual[isspecial.T]
            else:
                commit_loss = 0
                if for_warmup:
                    if vqmodel is not None:
                        # use features of a specific layer for kmeans warmup

                        # to B T H
                        if hasattr(self, 'tovq'):
                            projected_features = self.tovq(x).transpose(0, 1)
                        else:
                            projected_features = x.transpose(0, 1)

                        unquantized_features = projected_features[quantization_mask]

                        if type(vqmodel) == nn.ModuleList:
                            for onevq in vqmodel:
                                (quantized_features, categories, commit_loss, dist) = onevq(unquantized_features, self.minibz)
                        else:
                            (quantized_features, categories, commit_loss, dist) = vqmodel(unquantized_features, self.minibz)
                        # print(commit_loss)
                        # assert commit_loss.item() == 0
                else:
                    # to B T H
                    if hasattr(self, 'tovq'):
                        projected_features = self.tovq(x).transpose(0, 1)
                    else:
                        projected_features = x.transpose(0, 1)

                    unquantized_features = projected_features[quantization_mask]

                    # mean loss
                    if type(vqmodel) == nn.ModuleList:
                        exit('do not use multi vq')

                        multi_scale_features = []
                        for onevq in vqmodel:
                            (quantized_feature_vq, categories, commit_loss_vq, dist) = onevq(unquantized_features,
                                                                                          self.minibz)
                            commit_loss += commit_loss_vq

                            extra_content_annotations = torch.zeros_like(projected_features)
                            extra_content_annotations[quantization_mask] = quantized_feature_vq

                            multi_scale_features.append(extra_content_annotations)

                        commit_loss = commit_loss / len(vqmodel)
                        # M B T H
                        multi_scale_features = torch.stack(multi_scale_features, 0)

                        # attention
                        scale, bisze, tsize, hsize = multi_scale_features.size()
                        # M TB H
                        multi_scale_features = multi_scale_features.transpose(1, 2).reshape(scale, -1, hsize)

                        residual = x

                        if self.normalize_before:
                            x = self.vq_layer_norm(x)

                        # 1 TB H
                        x = x.view(1, -1, hsize)

                        x, _ = self.vqatt(
                            query=x,
                            key=multi_scale_features,
                            value=multi_scale_features,
                        )
                        x = self.dropout_module(x)

                        x = x.view(tsize, bisze, -1)

                        x = residual + x

                        if not self.normalize_before:
                            x = self.vq_layer_norm(x)

                    else:
                        (quantized_features, categories, commit_loss, dist) = vqmodel(unquantized_features, self.minibz)

                        # for analysis
                        # src_cats = x.new_full((x.size(1), x.size(0)), -1).long()
                        # src_cats[quantization_mask] = categories
                        # for c in src_cats:
                        #     print(c)
                        # exit()

                        isspecial = ~quantization_mask

                        residual = x

                        extra_content_annotations = torch.zeros_like(projected_features)
                        extra_content_annotations[quantization_mask] = quantized_features

                        # expand hidden size
                        extra_content_annotations = self.toemb(extra_content_annotations)
                        extra_content_annotations = self.vqdp(extra_content_annotations)
                        # T B H
                        extra_content_annotations = extra_content_annotations.transpose(0, 1)

                        x = residual + extra_content_annotations

                        if not self.normalize_before:
                            x = self.vq_layer_norm(x)

                        x[isspecial.T] = residual[isspecial.T]
        else:
            commit_loss = 0


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

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = residual + x

            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x

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

        # return x, attn, None
        return x, attn, None, commit_loss



class TransformerDecoderLayerMFFN(nn.Module):
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
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.use_rel_pos = getattr(args, "use_rel_pos", False)
        if self.use_rel_pos:
            self.self_attn = self.build_self_attention_rel_pos(self.embed_dim, args, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn)
        else:
            self.self_attn = self.build_self_attention(self.embed_dim, args, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn)

        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu")
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0)
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0)
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__)
        self.normalize_before = args.decoder_normalize_before

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)

        # added
        self.rezero = args.rezero

        if self.rezero:
            self.self_attn_layer_norm = None
            self.register_parameter('rezero_w1', nn.Parameter(torch.Tensor([0])))
        else:
            self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            # added
            if self.rezero:
                self.register_parameter('rezero_w2', nn.Parameter(torch.Tensor([0])))
                self.encoder_attn_layer_norm = None
            else:
                self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        # added
        self.headnum = args.decoder_attention_heads

        ffnkvnums = args.ffnkvnum
        self.ffn1 = nn.ModuleList()
        self.ffn2 = nn.ModuleList()
        for middim in ffnkvnums:
            fc1 = self.build_fc1(
                self.embed_dim // self.headnum, middim, self.quant_noise, self.quant_noise_block_size
            )
            fc2 = self.build_fc2(
                middim, self.embed_dim//self.headnum, self.quant_noise, self.quant_noise_block_size
            )
            self.ffn1.append(fc1)
            self.ffn2.append(fc2)

        # added
        if self.rezero:
            self.final_layer_norm = None
            self.register_parameter('rezero_w3', nn.Parameter(torch.Tensor([0])))
        else:
            self.final_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.need_attn = True

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
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


    def build_self_attention_rel_pos(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        return MultiheadAttentionWithRelPos(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            max_position_embeddings = args.max_relative_position,
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
            if self.self_attn_layer_norm is not None:
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
        # added
        if self.rezero:
            x = residual + x * self.rezero_w1
        else:
            x = residual + x

        if not self.normalize_before:
            if self.self_attn_layer_norm is not None:
                x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None:
            residual = x

            if self.normalize_before:
                if self.encoder_attn_layer_norm is not None:
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

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)

            # added
            if self.rezero:
                x = residual + x * self.rezero_w1
            else:
                x = residual + x

            if not self.normalize_before:
                if self.encoder_attn_layer_norm is not None:
                    x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            if self.final_layer_norm is not None:
                x = self.final_layer_norm(x)

        # x = self.activation_fn(self.fc1(x))
        # x = self.activation_dropout_module(x)
        # x = self.fc2(x)
        # x = self.dropout_module(x)

        x_splits = x.split(self.embed_dim//self.headnum, dim=-1)
        xs_ffn = []
        for i, x_head in enumerate(x_splits):
            x_head = self.activation_fn(self.ffn1[i](x_head))
            x_head = self.activation_dropout_module(x_head)
            x_head = self.ffn2[i](x_head)
            x_head = self.dropout_module(x_head)
            xs_ffn.append(x_head)
        x = torch.cat(xs_ffn, -1)

        # added
        if self.rezero:
            x = residual + x * self.rezero_w1
        else:
            x = residual + x

        if not self.normalize_before:
            if self.final_layer_norm is not None:
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

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
