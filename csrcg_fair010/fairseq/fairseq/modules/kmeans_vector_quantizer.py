# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from fairseq.modules import Fp32GroupNorm
import torch.nn.functional as F

#from torch.cuda.amp import autocast


def noop(*args, **kwargs):
    pass


def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device=device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device=device)

    return samples[indices]

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def orthgonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    n = t.shape[0]
    normed_codes = l2norm(t)
    identity = torch.eye(n, device = t.device)
    cosine_sim = torch.einsum('i d, j d -> i j', normed_codes, normed_codes)
    return ((cosine_sim - identity) ** 2).sum() / (n ** 2)


# added
class KmeansVectorQuantizerVQGAN2(nn.Module):
    def __init__(
            self, dim, num_vars, groups, combine_groups, vq_dim, time_first, gamma=0.25, ema=0.999, cosine=1, orth=0.0
    ):
        """Vector quantization using straight pass-through estimator (i.e. kmeans)

        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            gamma: commitment loss coefficient
        """
        super().__init__()

        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.vq_dim = vq_dim
        self.time_first = time_first

        assert (
                vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        self.var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        # self.embedding = nn.Parameter(
        # 0.01 * torch.randn(num_vars, num_groups, self.var_dim)
        # )

        # embedding = nn.Parameter(torch.randn(self.var_dim, num_vars))
        # embedding = nn.Parameter(torch.randn(num_vars, self.var_dim))

        self.use_cosine = cosine

        embedding = torch.randn(num_vars, self.var_dim)

        if ema:
            self.register_buffer("embedding", embedding)
        else:
            embedding = torch.empty(num_vars, self.var_dim)
            # nn.init.kaiming_uniform_(embedding)
            # embedding = F.normalize(embedding, dim=1)
            # if self.use_cosine == 0:
            #     nn.init.kaiming_uniform_(embedding)

            # taming-transformers#
            nn.init.uniform_(embedding, -1.0 / num_vars, 1.0 / num_vars)

            self.register_parameter("embedding", nn.Parameter(embedding))

        self.gamma = gamma
        self.mse_mean = nn.MSELoss(reduction="mean")

        self.orthreg = orth

        #
        self.ema = ema
        if self.ema:
            self.decay = ema

            kmeans_init = True

            # if not kmeans_init:
            #     embed = l2norm(uniform_init(codebook_size, dim))
            # else:
            #     embed = torch.zeros(codebook_size, dim)

            self.kmeans_iters = 10
            self.eps = 1e-5
            self.threshold_ema_dead_code = 0
            # self.sample_codebook_temp = sample_codebook_temp

            # self.sample_fn = sample_vectors_distributed if use_ddp else sample_vectors
            # self.all_reduce_fn = distributed.all_reduce if use_ddp else noop
            self.all_reduce_fn = noop
            self.sample_fn = sample_vectors

            self.register_buffer('initted', torch.Tensor([not kmeans_init]))
            self.register_buffer('cluster_size', torch.zeros(num_vars))
            self.register_buffer('embed_avg', embedding.clone())

    def _pass_grad(self, x, y):
        """Manually set gradient for backward pass.
        for y = f(x), ensure that during the backward pass,
        dL/dy = dL/dx regardless of f(x).
        Returns:
            y, with the gradient forced to be dL/dy = dL/dx.
        """
        return y.detach() + (x - x.detach())

    @property
    def expand_embedding(self):
        if self.combine_groups:
            return self.embedding.expand(self.num_vars, self.groups, self.var_dim)
        return self.embedding

    def forward_idx(self, x):
        res = self.forward(x, produce_targets=True)
        return res["x"], res["targets"]

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.ema == 0:
            return

        if self.initted:
            return

        with torch.no_grad():
            embed, cluster_size = kmeans(data, self.num_vars, self.kmeans_iters, use_cosine_sim=self.use_cosine,
                                     sample_fn=self.sample_fn, all_reduce_fn=self.all_reduce_fn)

        self.embedding.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

        print("kmeans ok")

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code
        if not torch.any(expired_codes):
            return

        # batch_samples = rearrange(batch_samples, '... d -> (...) d')
        batch_samples = batch_samples.view(-1, batch_samples.size(-1))

        # self.replace(batch_samples, mask = expired_codes)
        samples = F.normalize(batch_samples, dim=-1)
        self.embed.data[mask] = self.sample_fn(samples, mask.sum().item())

    def ema_update(self, embed_onehot, flatten, embed):
        # print('ema update', self.ema)

        # N*G K, N*G Z, Z K
        bins = embed_onehot.sum(0)
        self.all_reduce_fn(bins)

        ema_inplace(self.cluster_size, bins, self.decay)

        if self.use_cosine:
            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

        # Z K
        embed_sum = flatten.t() @ embed_onehot
        self.all_reduce_fn(embed_sum)

        if self.use_cosine:
            embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
            embed_normalized = F.normalize(embed_normalized, dim=-1)
            embed_normalized = torch.where(zero_mask[..., None], embed,
                                       embed_normalized)
            ema_inplace(self.embedding, embed_normalized, self.decay)
        else:
            ema_inplace(self.embed_avg, embed_sum.t(), self.decay)
            cluster_size = laplace_smoothing(self.cluster_size, self.num_vars, self.eps) * self.cluster_size.sum()
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embedding.data.copy_(embed_normalized)
        # self.expire_codes_(x)

    # @autocast(enabled=False)
    def forward(self, x):
        result = {"num_vars": self.num_vars}

        # N H
        nsz, fsz = x.shape
        dtype = x.dtype

        # to fp32
        # x = x.float()

        # print(self.embedding[0][:20])

        # N*G Z
        ze = x.view(-1, self.var_dim)

        if self.use_cosine:
            ze_norm = F.normalize(ze, dim=-1)
        else:
            ze_norm = ze

        self.init_embed_(ze_norm)

        # share embedding between groups
        if self.use_cosine:
            # K Z
            emb_norm = F.normalize(self.embedding, dim=1)
            d = - ze_norm @ emb_norm.t()
        else:
            emb_norm = self.embedding
            # N*G K
            d = (
                ze_norm.pow(2).sum(1, keepdim=True) - 2 * ze_norm @ emb_norm.t() + emb_norm.t().pow(2).sum(0, keepdim=True)
            )
        
        result["d"] = d

        # N*G
        idx = d.argmin(dim=1)

        # N*G Z
        zq = F.embedding(idx, self.embedding)

        x = self._pass_grad(ze, zq)
        # N H
        x = x.view(nsz, -1)

        result['idx'] = idx.clone().detach().view(-1, self.groups)

        # N H
        result["x"] = x

        # N*G K
        hard_x = F.one_hot(idx, self.num_vars).type(dtype)
        # hard_x = hard_x.view(nsz, self.groups, -1)
        result["codeuse"] = hard_x.sum(0)

        hard_probs = torch.mean(hard_x.float(), dim=0)
        result["code_perplexity"] = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        ze = ze.float()
        zq = zq.float()
        commitment_loss = self.mse_mean(ze, zq.detach())

        if self.ema > 0:
            latent_loss = 0
            # !!!
            if self.training:
                self.ema_update(hard_x, ze_norm, emb_norm)
        else:
            latent_loss = self.mse_mean(zq, ze.detach())

        result["kmloss"] = latent_loss + self.gamma * commitment_loss

        if self.training:
            if self.orthreg > 0:
                codebook = self.embedding
                # if self.orthogonal_reg_active_codes_only:
                    # only calculate orthogonal loss for the activated codes for this batch
                    # unique_code_ids = torch.unique(embed_ind)
                    # codebook = codebook[unique_code_ids]

                # num_codes = codebook.shape[0]
                # if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                #     rand_ids = torch.randperm(num_codes, device = device)[:self.orthogonal_reg_max_codes]
                #     codebook = codebook[rand_ids]

                orthogonal_reg_loss = orthgonal_loss_fn(codebook)
                result["orth_loss"] = orthogonal_reg_loss * self.orthreg

        return result


class KmeansVectorQuantizer(nn.Module):
    def __init__(
            self, dim, num_vars, groups, combine_groups, vq_dim, time_first, gamma=0.25
    ):
        """Vector quantization using straight pass-through estimator (i.e. kmeans)

        Args:
            dim: input dimension (channels)
            num_vars: number of quantized vectors per group
            groups: number of groups for vector quantization
            combine_groups: whether to use the vectors for all groups
            vq_dim: dimensionality of the resulting quantized vector
            time_first: if true, expect input in BxTxC format, otherwise in BxCxT
            gamma: commitment loss coefficient
        """
        super().__init__()

        self.groups = groups
        self.combine_groups = combine_groups
        self.input_dim = dim
        self.num_vars = num_vars
        self.vq_dim = vq_dim
        self.time_first = time_first

        assert (
                vq_dim % groups == 0
        ), f"dim {vq_dim} must be divisible by groups {groups} for concatenation"

        self.var_dim = vq_dim // groups
        num_groups = groups if not combine_groups else 1

        self.embedding = nn.Parameter(
            0.01 * torch.randn(num_vars, num_groups, self.var_dim)
        )
        self.projection = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, groups=groups, bias=False),
            Fp32GroupNorm(groups, dim),
        )
        self.gamma = gamma
        self.mse_mean = nn.MSELoss(reduction="mean")

    def _pass_grad(self, x, y):
        """Manually set gradient for backward pass.
        for y = f(x), ensure that during the backward pass,
        dL/dy = dL/dx regardless of f(x).
        Returns:
            y, with the gradient forced to be dL/dy = dL/dx.
        """

        return y.detach() + (x - x.detach())

    @property
    def expand_embedding(self):
        if self.combine_groups:
            return self.embedding.expand(self.num_vars, self.groups, self.var_dim)
        return self.embedding

    def forward_idx(self, x):
        res = self.forward(x, produce_targets=True)
        return res["x"], res["targets"]

    def forward(self, x, produce_targets=False):

        result = {"num_vars": self.num_vars}

        if self.time_first:
            x = x.transpose(1, 2)

        bsz, fsz, tsz = x.shape

        ze = self.projection(x)
        ze_ = ze.view(bsz, self.groups, self.var_dim, tsz).permute(0, 3, 1, 2)
        d = (
            (ze_.unsqueeze(0) - self.expand_embedding.unsqueeze(1).unsqueeze(1))
                .view(self.num_vars, bsz, tsz, self.groups, -1)
                .norm(dim=-1, p=2)
        )
        idx = d.argmin(dim=0)
        zq = (
            torch.stack(
                [
                    self.expand_embedding[idx[..., group], group]
                    for group in range(self.groups)
                ],
                dim=-2,
            )
                .view(bsz, tsz, self.groups * self.var_dim)
                .permute(0, 2, 1)
        )
        assert ze.shape == zq.shape, (ze.shape, zq.shape)
        x = self._pass_grad(ze, zq)

        hard_x = (
            idx.new_zeros(bsz * tsz * self.groups, self.num_vars)
                .scatter_(-1, idx.view(-1, 1), 1.0)
                .view(bsz * tsz, self.groups, -1)
        )
        hard_probs = torch.mean(hard_x.float(), dim=0)
        result["code_perplexity"] = torch.exp(
            -torch.sum(hard_probs * torch.log(hard_probs + 1e-7), dim=-1)
        ).sum()

        if produce_targets:
            result["targets"] = idx

        if self.time_first:
            x = x.transpose(1, 2)  # BCT -> BTC
        result["x"] = x

        ze = ze.float()
        zq = zq.float()
        latent_loss = self.mse_mean(zq, ze.detach())
        commitment_loss = self.mse_mean(ze, zq.detach())

        result["kmeans_loss"] = latent_loss + self.gamma * commitment_loss

        return result


def kmeans(samples, num_clusters, num_iters=10, use_cosine_sim=False,
           sample_fn=sample_vectors, all_reduce_fn=noop):
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ means.t()
        else:
            # diffs = rearrange(samples, 'n d -> n () d') \
            #         - rearrange(means, 'c d -> () c d')
            dists = samples.pow(2).sum(1, keepdim=True) - 2 * samples @ means.t() + means.t().pow(2).sum(0, keepdim=True)

        buckets = torch.argmax(dists, dim=-1)
        bins = torch.bincount(buckets, minlength=num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_clusters, dim, dtype=dtype)
        # new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d = dim), samples)

        new_means.scatter_add_(0, buckets.unsqueeze(1).expand(-1, dim), samples)

        new_means = new_means / bins_min_clamped[..., None]
        all_reduce_fn(new_means)

        if use_cosine_sim:
            # new_means = l2norm(new_means)
            new_means = F.normalize(new_means, dim=-1)

        means = torch.where(zero_mask[..., None], means, new_means)

    return means, bins
