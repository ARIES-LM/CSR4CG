# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

import torch
import torch.nn as nn

from fairseq import utils
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
    GradMultiply,
    VectorQuantize
)
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.fairseq_dropout import FairseqDropout


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
            self,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 3072,
            num_attention_heads: int = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = 'relu',
            export: bool = False,
            q_noise: float = 0.0,
            qn_block_size: int = 8,
            init_fn: Callable = None,
    ) -> None:
        super().__init__()

        if init_fn is not None:
            init_fn()

        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.activation_dropout_module = FairseqDropout(activation_dropout, module_name=self.__class__.__name__)

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = self.build_self_attention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embedding_dim,
            ffn_embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.fc2 = self.build_fc2(
            ffn_embedding_dim,
            self.embedding_dim,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), q_noise, qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), q_noise, qn_block_size
        )

    def build_self_attention(
            self,
            embed_dim,
            num_attention_heads,
            dropout,
            self_attention,
            q_noise,
            qn_block_size,
    ):
        return MultiheadAttention(
            embed_dim,
            num_attention_heads,
            dropout=dropout,
            self_attention=True,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            **kwargs
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn


# added
class TransformerSentenceEncoderLayerVQ(TransformerSentenceEncoderLayer):
    def __init__(
            self,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 3072,
            num_attention_heads: int = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = 'relu',
            export: bool = False,
            q_noise: float = 0.0,
            qn_block_size: int = 8,
            init_fn: Callable = None,
            vqdim=256, k=128, vq_share=True, vq_decay=0.97, vq_commitment=1.0, vq_wait_steps=1, vq_observe_steps=1,
            kmbatch_multiplier=10
    ) -> None:
        super().__init__(embedding_dim,
                         ffn_embedding_dim,
                         num_attention_heads,
                         dropout,
                         attention_dropout,
                         activation_dropout,
                         activation_fn,
                         export,
                         q_noise,
                         qn_block_size,
                         init_fn, )

        # added
        self.minibz = 10 * k
        embed_dim = embedding_dim

        self.tovq = nn.Linear(embed_dim, vqdim, bias=False)

        if not vq_share:
            self.vq = VectorQuantize(
                dim=vqdim,
                n_embed=k,
                decay=vq_decay, # ema update
                commitment=vq_commitment, # loss weight
                wait_steps=vq_wait_steps,
                observe_steps=vq_observe_steps, # kmeans initialization
                coreset_size_multiplier=kmbatch_multiplier
            )

        self.vqshare = vq_share

        self.vqdp = FairseqDropout(dropout, module_name=self.__class__.__name__)

        self.toemb = nn.Linear(vqdim, embed_dim, bias=False)
        self.vq_layer_norm = LayerNorm(embed_dim, export=export)

        # histogram
        self.availability = {i:0 for i in range(k)}
        self.print_code_usage = False

    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            quantization_mask=None,
            for_warmup=False, vqmodel=None
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        # added
        if not self.vqshare:
            # to B T H
            projected_features = self.tovq(x).transpose(0, 1)
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
                # T B H
                extra_content_annotations = extra_content_annotations.transpose(0, 1)

                # expand hidden size
                extra_content_annotations = self.toemb(extra_content_annotations)
                extra_content_annotations = self.vqdp(extra_content_annotations)
                x = residual + extra_content_annotations
                x = self.vq_layer_norm(x)

                x[isspecial.T] = residual[isspecial.T]
        else:
            commit_loss = 0
            if for_warmup:
                if vqmodel is not None:
                    # use features of a specific layer for kmeans warmup

                    # to B T H
                    projected_features = self.tovq(x).transpose(0, 1)
                    unquantized_features = projected_features[quantization_mask]
                    (quantized_features, categories, commit_loss, dist) = vqmodel(unquantized_features, self.minibz)
                    # assert commit_loss.item() == 0

            else:
                # to B T H
                projected_features = self.tovq(x).transpose(0, 1)
                unquantized_features = projected_features[quantization_mask]

                # mean loss
                (quantized_features, categories, commit_loss, dist) = vqmodel(unquantized_features, self.minibz)

                assert dist is not None

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
                x = self.vq_layer_norm(x)

                x[isspecial.T] = residual[isspecial.T]

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.final_layer_norm(x)

        # for analysis
        if self.print_code_usage:
            for ki in self.availability:
                code_i_use = (categories == ki).sum().item()
                self.availability[ki] += code_i_use

        return x, attn, commit_loss


class TransformerSentenceEncoderLayerAMP(TransformerSentenceEncoderLayer):
    def __init__(
            self,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 3072,
            num_attention_heads: int = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            activation_fn: str = 'relu',
            export: bool = False,
            q_noise: float = 0.0,
            qn_block_size: int = 8,
            init_fn: Callable = None,
            packdim=256, qtopk=0
    ) -> None:
        super().__init__(embedding_dim,
                         ffn_embedding_dim,
                         num_attention_heads,
                         dropout,
                         attention_dropout,
                         activation_dropout,
                         activation_fn,
                         export,
                         q_noise,
                         qn_block_size,
                         init_fn, )

        # print(self.embedding_dim, packdim, num_attention_heads)
        headnum = 8
        self.pack_attn = self.build_pack_attentionv2(
            packdim,
            self.embedding_dim,
            self.embedding_dim,
            headnum,
            dropout=attention_dropout,
            self_attention=False,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.pack_attn_layer_norm = LayerNorm(packdim, export=export)


        self.pack_self_attn = self.build_pack_attentionv2(
            packdim, packdim, packdim,
            headnum,
            dropout=attention_dropout,
            self_attention=False,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.pack_self_attn_layer_norm = LayerNorm(packdim, export=export)

        self.mpffn = 0
        # args.mpffn
        # if self.mpffn:
        #     self.mpfc1 = self.build_fc1(
        #         self.packdim,
        #         args.encoder_ffn_embed_dim,
        #         self.quant_noise,
        #         self.quant_noise_block_size,
        #     )
        #     self.mpfc2 = self.build_fc2(
        #         args.packdim,
        #         self.embed_dim,
        #         self.quant_noise,
        #         self.quant_noise_block_size,
        #     )
        #     self.mpffn_layer_norm = LayerNorm(packdim, export=export)

        self.unpack_attn = self.build_pack_attentionv2(
            self.embedding_dim,
            packdim, packdim,
            headnum,
            dropout=attention_dropout,
            self_attention=False,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )
        self.unpack_layer_norm = LayerNorm(self.embedding_dim, export=export)

        self.batchmp = False

        self.fullatt = False
        self.ablation = False
        self.topk = qtopk

    def build_pack_attention(
            self,
            embed_dim,
            num_attention_heads,
            dropout,
            self_attention,
            q_noise,
            qn_block_size,
    ):
        return MultiheadAttention(
            embed_dim,
            num_attention_heads,
            dropout=dropout,
            self_attention=False,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def build_pack_attentionv2(
            self,
            qdim,
            kdim,
            vdim,
            num_attention_heads,
            dropout,
            self_attention,
            q_noise,
            qn_block_size,
    ):
        return MultiheadAttention(
            qdim,
            num_attention_heads,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            self_attention=False,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def residual_connection(self, x, y):
        return x + y

    def forward(
            self,
            x: torch.Tensor,
            self_attn_mask: Optional[torch.Tensor] = None,
            self_attn_padding_mask: Optional[torch.Tensor] = None,
            packed_x=None
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        residual = x
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        if self.training and self.batchmp:
            psize, bsize, _ = packed_x.size()

            # pack
            residual_px = packed_x

            # P B H
            packed_x, _ = self.pack_attn(
                query=packed_x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
            )
            packed_x = self.dropout_module(packed_x)
            packed_x = self.residual_connection(packed_x, residual_px)
            packed_x = self.pack_attn_layer_norm(packed_x)

            # exchange
            if not self.ablation:
                residual_px = packed_x

                # knn
                with torch.no_grad():
                    distance = torch.cdist(packed_x, packed_x)
                    # P B k
                    _, smallk_index = torch.topk(distance, self.topk, dim=2, largest=False)

                total_packed_x = packed_x.unsqueeze(1).expand(-1, bsize, -1, -1)

                selected = torch.gather(total_packed_x, dim=2,
                                        index=smallk_index.unsqueeze(-1).expand(-1, -1, -1, packed_x.size(-1)))

                # if self.fullatt:
                #     # PB 1 H
                #     packed_x = packed_x.reshape(-1, 1, packed_x.size(-1))

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

                if self.topk > 0:
                    # 1 PB H
                    packed_x = packed_x.reshape(-1, 1, packed_x.size(-1)).transpose(0, 1)
                    # k PB H
                    selected = selected.reshape(-1, selected.size(2), selected.size(3)).transpose(0, 1)
                    # print(packed_x.size(), selected.size())

                    # 1 PB H
                    packed_x, _ = self.pack_self_attn(
                        query=packed_x,
                        key=selected,
                        value=selected,
                    )
                    packed_x = packed_x.reshape(psize, bsize, -1)
                    # exit('ok')
                else:
                    # B P H
                    packed_x = packed_x.transpose(0, 1)

                    packed_x, _ = self.pack_self_attn(
                        query=packed_x,
                        key=packed_x,
                        value=packed_x,
                    )
                    packed_x = packed_x.transpose(0, 1)

                # P B H
                # if self.fullatt:
                #     packed_x = packed_x.reshape(psize, bsize, -1)
                # else:
                #     packed_x = packed_x.transpose(0, 1)

                packed_x = self.dropout_module(packed_x)
                packed_x = self.residual_connection(packed_x, residual_px)
                packed_x = self.pack_self_attn_layer_norm(packed_x)

                if self.mpffn:
                    residual_px = packed_x
                    packed_x = self.activation_fn(self.mpfc1(packed_x))
                    packed_x = self.activation_dropout_module(packed_x)
                    packed_x = self.mpfc2(packed_x)
                    packed_x = self.dropout_module(packed_x)
                    packed_x = self.residual_connection(packed_x, residual_px)
                    packed_x = self.mpffn_layer_norm(packed_x)

            # unpack
            residual = x
            x, _ = self.unpack_attn(
                query=x,
                key=packed_x,
                value=packed_x,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            x = self.unpack_layer_norm(x)

            # added
            x = GradMultiply.apply(x, 1/2)


        residual = x
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        x = self.final_layer_norm(x)

        if self.training and self.batchmp:
            return x, packed_x, attn

        return x, attn
