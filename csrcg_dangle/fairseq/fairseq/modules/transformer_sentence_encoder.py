# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn as nn
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    MultiheadAttention,
    PositionalEmbedding,
    TransformerSentenceEncoderLayer,
    TransformerSentenceEncoderLayerAMP,
    TransformerSentenceEncoderLayerVQ,
    GradMultiply,
    VectorQuantize
)
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
import math


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        module.q_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.k_proj.weight.data.normal_(mean=0.0, std=0.02)
        module.v_proj.weight.data.normal_(mean=0.0, std=0.02)


class TransformerSentenceEncoder(nn.Module):
    """
    Implementation for a Bi-directional Transformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape T x B x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
            self,
            padding_idx: int,
            vocab_size: int,
            num_encoder_layers: int = 6,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 3072,
            num_attention_heads: int = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            layerdrop: float = 0.0,
            max_seq_len: int = 256,
            num_segments: int = 2,
            use_position_embeddings: bool = True,
            offset_positions_by_padding: bool = True,
            encoder_normalize_before: bool = False,
            apply_bert_init: bool = False,
            activation_fn: str = "relu",
            learned_pos_embedding: bool = True,
            embed_scale: float = None,
            freeze_embeddings: bool = False,
            n_trans_layers_to_freeze: int = 0,
            export: bool = False,
            traceable: bool = False,
            q_noise: float = 0.0,
            qn_block_size: int = 8,
    ) -> None:

        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout_module = FairseqDropout(dropout, module_name=self.__class__.__name__)
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings
        self.apply_bert_init = apply_bert_init
        self.learned_pos_embedding = learned_pos_embedding
        self.traceable = traceable
        self.tpu = False  # whether we're on TPU

        self.embed_tokens = self.build_embedding(
            self.vocab_size, self.embedding_dim, self.padding_idx
        )
        self.embed_scale = embed_scale

        # print(self.embed_scale)
        # None
        # exit()

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        self.segment_embeddings = (
            nn.Embedding(self.num_segments, self.embedding_dim, padding_idx=None)
            if self.num_segments > 0
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                self.max_seq_len,
                self.embedding_dim,
                padding_idx=(self.padding_idx if offset_positions_by_padding else None),
                learned=self.learned_pos_embedding,
            )
            if self.use_position_embeddings
            else None
        )

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend([
            self.build_transformer_sentence_encoder_layer(
                embedding_dim=self.embedding_dim,
                ffn_embedding_dim=ffn_embedding_dim,
                num_attention_heads=num_attention_heads,
                dropout=self.dropout_module.p,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn,
                export=export,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
                lix=lix
            )
            for lix in range(num_encoder_layers)
        ])

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        # Apply initialization of model params after building the model

        if self.apply_bert_init:
            self.apply(init_bert_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            freeze_module_params(self.embed_tokens)
            freeze_module_params(self.segment_embeddings)
            freeze_module_params(self.embed_positions)
            freeze_module_params(self.emb_layer_norm)

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_transformer_sentence_encoder_layer(
            self,
            embedding_dim,
            ffn_embedding_dim,
            num_attention_heads,
            dropout,
            attention_dropout,
            activation_dropout,
            activation_fn,
            export,
            q_noise,
            qn_block_size,
            lix=None
    ):
        return TransformerSentenceEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
        )

    def prepare_for_tpu_(self, **kwargs):
        self.tpu = True

    def forward(
            self,
            tokens: torch.Tensor,
            tgt_embed: torch.Tensor = None,
            tgt_padding_mask: torch.Tensor = None,
            segment_labels: torch.Tensor = None,
            last_state_only: bool = False,
            positions: Optional[torch.Tensor] = None,
            **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # compute padding mask. This is needed for multi-head attention
        padding_mask = tokens.eq(self.padding_idx)
        # if not self.traceable and not self.tpu and not padding_mask.any():
        # padding_mask = None

        x = self.embed_tokens(tokens)

        if self.embed_scale is not None:
            x *= self.embed_scale

        if self.embed_positions is not None:
            x += self.embed_positions(tokens, positions=positions)

        if self.segment_embeddings is not None and segment_labels is not None:
            x += self.segment_embeddings(segment_labels)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation
        if padding_mask is not None:
            x *= 1 - padding_mask.unsqueeze(-1).type_as(x)

        # added for dangle
        if tgt_embed is not None:
            # print("tgt_embed shape : "+str(tgt_embed.shape))
            x = torch.cat([x, tgt_embed], dim=1)
            padding_mask = torch.cat([padding_mask, tgt_padding_mask], dim=1)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x, _ = layer(x, self_attn_padding_mask=padding_mask)
            if not last_state_only:
                inner_states.append(x)

        sentence_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep


# added
class TransformerSentenceEncoderVQ(TransformerSentenceEncoder):
    def __init__(
            self,
            padding_idx: int,
            vocab_size: int,
            num_encoder_layers: int = 6,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 3072,
            num_attention_heads: int = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            layerdrop: float = 0.0,
            max_seq_len: int = 256,
            num_segments: int = 2,
            use_position_embeddings: bool = True,
            offset_positions_by_padding: bool = True,
            encoder_normalize_before: bool = False,
            apply_bert_init: bool = False,
            activation_fn: str = "relu",
            learned_pos_embedding: bool = True,
            embed_scale: float = None,
            freeze_embeddings: bool = False,
            n_trans_layers_to_freeze: int = 0,
            export: bool = False,
            traceable: bool = False,
            q_noise: float = 0.0,
            qn_block_size: int = 8,
            vqdim=256, k=128, vq_share=True, vq_decay=0.97, vq_commitment=1.0,
            vq_wait_steps=1, vq_observe_steps=1, kmbatch_multiplier=10, vqlayers=0
    ) -> None:

        self.layer4kms = vqlayers
        self.vq_layers = vqlayers
        self.vqdim =vqdim
        self.k = k
        self.vqshare = vq_share
        self.vq_decay = vq_decay
        self.vq_commitment = vq_commitment
        self.vq_wait_steps = vq_wait_steps
        self.vq_observe_steps = vq_observe_steps
        self.kmbatch_multiplier = kmbatch_multiplier

        super().__init__(padding_idx,
                         vocab_size,
                         num_encoder_layers,
                         embedding_dim,
                         ffn_embedding_dim,
                         num_attention_heads,
                         dropout,
                         attention_dropout,
                         activation_dropout,
                         layerdrop,
                         max_seq_len,
                         num_segments,
                         use_position_embeddings,
                         offset_positions_by_padding,
                         encoder_normalize_before,
                         apply_bert_init,
                         activation_fn,
                         learned_pos_embedding,
                         embed_scale,
                         freeze_embeddings,
                         n_trans_layers_to_freeze,
                         export,
                         traceable,
                         q_noise,
                         qn_block_size)


    def build_transformer_sentence_encoder_layer(
            self,
            embedding_dim,
            ffn_embedding_dim,
            num_attention_heads,
            dropout,
            attention_dropout,
            activation_dropout,
            activation_fn,
            export,
            q_noise,
            qn_block_size,
            lix=None
    ):
        if lix >= self.vq_layers:
            return TransformerSentenceEncoderLayerVQ(
                embedding_dim=embedding_dim,
                ffn_embedding_dim=ffn_embedding_dim,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn,
                export=export,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
                vqdim=self.vqdim, k=self.k, vq_share=self.vqshare,
                vq_decay=self.vq_decay, vq_commitment=self.vq_commitment,
                vq_wait_steps=self.vq_wait_steps, vq_observe_steps=self.vq_observe_steps,
                kmbatch_multiplier=self.kmbatch_multiplier
            )
        else:
            return TransformerSentenceEncoderLayer(
                embedding_dim=embedding_dim,
                ffn_embedding_dim=ffn_embedding_dim,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn,
                export=export,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
            )

    def forward(
            self,
            tokens: torch.Tensor,
            tgt_embed: torch.Tensor = None,
            tgt_padding_mask: torch.Tensor = None,
            segment_labels: torch.Tensor = None,
            last_state_only: bool = False,
            positions: Optional[torch.Tensor] = None,
            vq=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # compute padding mask. This is needed for multi-head attention
        padding_mask = tokens.eq(self.padding_idx)
        # if not self.traceable and not self.tpu and not padding_mask.any():
        # padding_mask = None

        x = self.embed_tokens(tokens)

        if self.embed_scale is not None:
            x *= self.embed_scale

        if self.embed_positions is not None:
            x += self.embed_positions(tokens, positions=positions)

        if self.segment_embeddings is not None and segment_labels is not None:
            x += self.segment_embeddings(segment_labels)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation
        if padding_mask is not None:
            x *= 1 - padding_mask.unsqueeze(-1).type_as(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        # added
        # todo check
        isbos = tokens.eq(0)
        iseos = tokens.eq(2)
        isspecial = padding_mask | isbos | iseos
        # B T
        quantization_mask = ~ isspecial

        cmlosses = []

        for lix, layer in enumerate(self.layers):
            if lix >= self.vq_layers:
                # added
                if self.vqshare:
                    if self.training and vq.observe_steps_remaining != 0:
                        if lix == self.layer4kms:
                            x, _, cmloss = layer(x, self_attn_padding_mask=padding_mask,
                                                 quantization_mask=quantization_mask,
                                                 for_warmup=True, vqmodel=vq)
                        else:
                            x, _, cmloss = layer(x, self_attn_padding_mask=padding_mask,
                                                 quantization_mask=quantization_mask,
                                                 for_warmup=True)
                    else:
                        x, _, cmloss = layer(x, self_attn_padding_mask=padding_mask,
                                             quantization_mask=quantization_mask,
                                             for_warmup=False, vqmodel=vq)
                else:
                    x, _, cmloss = layer(x, self_attn_padding_mask=padding_mask, quantization_mask=quantization_mask)

                cmlosses.append(cmloss)

            else:
                x, _ = layer(x, self_attn_padding_mask=padding_mask)


            if not last_state_only:
                inner_states.append(x)

        # added
        cmloss = sum(cmlosses)/len(cmlosses)

        sentence_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return (torch.stack(inner_states), cmloss), sentence_rep
        else:
            return (inner_states, cmloss), sentence_rep



# added
class TransformerSentenceEncoderAMP(TransformerSentenceEncoder):
    def __init__(
            self,
            padding_idx: int,
            vocab_size: int,
            num_encoder_layers: int = 6,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 3072,
            num_attention_heads: int = 8,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            layerdrop: float = 0.0,
            max_seq_len: int = 256,
            num_segments: int = 2,
            use_position_embeddings: bool = True,
            offset_positions_by_padding: bool = True,
            encoder_normalize_before: bool = False,
            apply_bert_init: bool = False,
            activation_fn: str = "relu",
            learned_pos_embedding: bool = True,
            embed_scale: float = None,
            freeze_embeddings: bool = False,
            n_trans_layers_to_freeze: int = 0,
            export: bool = False,
            traceable: bool = False,
            q_noise: float = 0.0,
            qn_block_size: int = 8,
            packnum=4,
            amplayers=None,
            packdim=256, sharepack=1, qtopk=0
    ) -> None:
        self.amplayers = int(amplayers)
        self.packdim = packdim
        self.qtopk = qtopk

        super().__init__(padding_idx,
                         vocab_size,
                         num_encoder_layers,
                         embedding_dim,
                         ffn_embedding_dim,
                         num_attention_heads,
                         dropout,
                         attention_dropout,
                         activation_dropout,
                         layerdrop,
                         max_seq_len,
                         num_segments,
                         use_position_embeddings,
                         offset_positions_by_padding,
                         encoder_normalize_before,
                         apply_bert_init,
                         activation_fn,
                         learned_pos_embedding,
                         embed_scale,
                         freeze_embeddings,
                         n_trans_layers_to_freeze,
                         export,
                         traceable,
                         q_noise,
                         qn_block_size)

        self.pack_num = packnum
        self.batchmp = False

        if apply_bert_init:
            # roberta init
            self.register_parameter('pack_emb', nn.Parameter(torch.normal(0.0, 0.02,
                                                                      size=(packnum, packdim))))
            self.pack_scale = None
        else:
            # fairseq init
            self.register_parameter('pack_emb', nn.Parameter(torch.normal(0.0, packdim ** -0.5,
                                                                          size=(packnum, packdim))))
            self.pack_scale = math.sqrt(packdim)

        self.packemb_layer_norm = LayerNorm(packdim, export=export)
        # print('self.emb_layer_norm', self.emb_layer_norm)
        # yes

        # sharing
        if sharepack:
            for lix, layer in enumerate(self.layers):
                if lix > self.amplayers:
                    layer.pack_attn = self.layers[self.amplayers].pack_attn
                    layer.pack_self_attn = self.layers[self.amplayers].pack_self_attn
                    layer.unpack_attn = self.layers[self.amplayers].unpack_attn

                    layer.pack_attn_layer_norm = self.layers[self.amplayers].pack_attn_layer_norm
                    layer.pack_self_attn_layer_norm = self.layers[self.amplayers].pack_self_attn_layer_norm
                    layer.unpack_layer_norm = self.layers[self.amplayers].unpack_layer_norm

    # added
    def set_batchmp(self, flag):
        self.batchmp = flag
        for l in self.layers:
            l.batchmp = flag

    def build_transformer_sentence_encoder_layer(
            self,
            embedding_dim,
            ffn_embedding_dim,
            num_attention_heads,
            dropout,
            attention_dropout,
            activation_dropout,
            activation_fn,
            export,
            q_noise,
            qn_block_size,
            lix=None
    ):
        if lix >= self.amplayers:
            return TransformerSentenceEncoderLayerAMP(
                embedding_dim=embedding_dim,
                ffn_embedding_dim=ffn_embedding_dim,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn,
                export=export,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
                packdim=self.packdim, qtopk=self.qtopk
            )
        else:
            return TransformerSentenceEncoderLayer(
                embedding_dim=embedding_dim,
                ffn_embedding_dim=ffn_embedding_dim,
                num_attention_heads=num_attention_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                activation_fn=activation_fn,
                export=export,
                q_noise=q_noise,
                qn_block_size=qn_block_size,
            )

    def forward(
            self,
            tokens: torch.Tensor,
            tgt_embed: torch.Tensor = None,
            tgt_padding_mask: torch.Tensor = None,
            segment_labels: torch.Tensor = None,
            last_state_only: bool = False,
            positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # compute padding mask. This is needed for multi-head attention
        padding_mask = tokens.eq(self.padding_idx)
        # if not self.traceable and not self.tpu and not padding_mask.any():
        # padding_mask = None

        x = self.embed_tokens(tokens)

        if self.embed_scale is not None:
            x *= self.embed_scale

        if self.embed_positions is not None:
            x += self.embed_positions(tokens, positions=positions)

        if self.segment_embeddings is not None and segment_labels is not None:
            x += self.segment_embeddings(segment_labels)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation
        if padding_mask is not None:
            x *= 1 - padding_mask.unsqueeze(-1).type_as(x)

        # added for dangle
        if tgt_embed is not None:
            # print("tgt_embed shape : "+str(tgt_embed.shape))
            x = torch.cat([x, tgt_embed], dim=1)
            padding_mask = torch.cat([padding_mask, tgt_padding_mask], dim=1)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        # added
        pack_emb = None
        if self.training and self.batchmp:
            # if self.lendiv:
            #     pack_num = src_tokens.size(-1) // self.lendiv
            # else:
            pack_num = self.pack_num

            # pack_ids = [self.dictionary.index('pack_{}'.format(i)) for i in range(pack_num)]
            # print(pack_ids)
            # exit()
            # pack_ids = tokens.new_tensor(pack_ids)
            # B P
            # pack_ids = pack_ids.unsqueeze(0).expand(src_tokens.size(0), -1)

            # pack_emb = self.embed_tokens(pack_ids)

            # B P H
            pack_emb = self.pack_emb.unsqueeze(0).expand(tokens.size(0), -1, -1)

            pack_emb = GradMultiply.apply(pack_emb, 2.0)

            if self.pack_scale is not None:
                pack_emb = pack_emb * self.pack_scale

            # print(pack_emb.size())
            # exit()

            # if self.embed_scale is not None:
            #     pack_emb *= self.embed_scale

            # if self.packposi:
            #     pack_emb = pack_emb + self.embed_positions(pack_ids)

            if self.quant_noise is not None:
                pack_emb = self.quant_noise(pack_emb)

            if self.emb_layer_norm is not None:
                pack_emb = self.packemb_layer_norm(pack_emb)

            pack_emb = self.dropout_module(pack_emb)
            # P B H
            pack_emb = pack_emb.transpose(0, 1)

        for lix, layer in enumerate(self.layers):
            # x, _ = layer(x, self_attn_padding_mask=padding_mask)

            if pack_emb is not None and lix >= self.amplayers:
                x, pack_emb, _ = layer(x, self_attn_padding_mask=padding_mask, packed_x=pack_emb)
            else:
                x, _ = layer(x, self_attn_padding_mask=padding_mask)

            if not last_state_only:
                inner_states.append(x)

        sentence_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep
