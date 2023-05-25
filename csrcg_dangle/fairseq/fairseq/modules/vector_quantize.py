import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# from clusopt_core.cluster import Streamkm
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
import torch.distributed as distributed

def ema_inplace(moving_avg, new, decay):
    moving_avg.data.mul_(decay).add_(new, alpha=(1 - decay))


def laplace_smoothing(x, n_categories, eps=1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)


class VectorQuantize(nn.Module):
    # Based on: https://github.com/lucidrains/vector-quantize-pytorch
    def __init__(
        self,
        dim,
        n_embed,
        decay=0.8,
        commitment=1.0,
        eps=1e-5,
        wait_steps=0,
        observe_steps=1245,
        coreset_size_multiplier=10,
        usecosine=0,
    ):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.commitment = commitment

        self.ema = 1

        embed = torch.randn(dim, n_embed)

        self.register_buffer("embed", nn.Parameter(embed))
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", nn.Parameter(embed.clone()))

        self.wait_steps_remaining = wait_steps
        self.observe_steps_remaining = observe_steps

        # added
        self.batchsize = n_embed * coreset_size_multiplier
        self.clustering_model = MiniBatchKMeans(n_clusters=n_embed, random_state=42,
                                                batch_size=n_embed*coreset_size_multiplier)

        self.usecosine = usecosine
        # print('cosine', self.usecosine)
        # self.clustering_model = Streamkm(
        #     coresetsize=n_embed * coreset_size_multiplier,
        #     length=1500000,
        #     seed=42,
        # )

        self.data_chunks = []

    def stream_cluster(self, input, expected_num_tokens=None):

        if self.wait_steps_remaining > 0:

            if self.wait_steps_remaining % 100 == 0:
                print('wait_steps_remaining', self.wait_steps_remaining)

            self.wait_steps_remaining -= 1
            # self.data_chunks.clear()
            return

        # added
        if torch.cuda.device_count() > 1:
            # * H
            cuda_device = f'cuda:{distributed.get_rank()}'
            local_size = torch.tensor(input.size()[0], device=cuda_device)

            ws = distributed.get_world_size()
            all_sizes = [torch.zeros_like(local_size) for _ in range(ws)]
            distributed.all_gather(all_sizes, local_size)

            # print(all_sizes)
            max_size = max(all_sizes)
            # print(max_size)

            size_diff = max_size.item() - local_size.item()
            if size_diff:
                padding = input.new_zeros(size_diff, input.size(-1))
                # print(padding.size(), packed_x.size())
                pad_packed_x = torch.cat((input, padding), 0)
            else:
                pad_packed_x = input

            all_qs_padded = [torch.zeros_like(pad_packed_x) for _ in range(ws)]
            distributed.all_gather(all_qs_padded, pad_packed_x)
            all_qs = []
            for q, size in zip(all_qs_padded, all_sizes):
                all_qs.append(q[:size])

            # for gradient
            # all_qs[dist.get_rank()] = packed_x

            input = torch.cat(all_qs, 0)


        input_np = input.detach().cpu().numpy()
        assert len(input.shape) == 2

        self.data_chunks.append(input_np)

        if (
            expected_num_tokens is not None
            and sum([chunk.shape[0] for chunk in self.data_chunks])
            < expected_num_tokens
        ):
            return  # This is not the last sub-batch.

        if self.observe_steps_remaining % 100 == 0:
            print('kmeans warmup steps remain', self.observe_steps_remaining)

        self.observe_steps_remaining -= 1
        input_np = np.concatenate(self.data_chunks, axis=0)

        self.data_chunks.clear()

        # self.clustering_model.partial_fit(input_np)

        # added
        self.clustering_model = self.clustering_model.partial_fit(input_np)

        # print('self.embed.dtype, input.dtype', self.embed.dtype, input.dtype)

        if self.observe_steps_remaining == 0:
            print("Initializing vq clusters (this may take a while)...")

            # clusters, _ = self.clustering_model.get_final_clusters(
            #     self.n_embed, seed=42
            # )

            # added
            # K H
            clusters = self.clustering_model.cluster_centers_

            new_embed = torch.tensor(
                clusters.T, dtype=self.embed.dtype, device=self.embed.device
            )

            # added
            if self.usecosine:
                # H K
                new_embed = F.normalize(new_embed, dim=0)

            self.embed.copy_(new_embed)
            # Don't set initial cluster sizes to zero! If a cluster is rare,
            # embed_avg will be undergoing exponential decay until it's seen for
            # the first time. If cluster_size is zero, this will lead to *embed*
            # also undergoing exponential decay towards the origin before the
            # cluster is ever encountered. Initializing to 1.0 will instead will
            # instead leave embed in place for many iterations, up until
            # cluster_size finally decays to near-zero.
            self.cluster_size.fill_(1.0)
            self.embed_avg.copy_(new_embed)

    def forward(self, input, expected_num_tokens=None):
        # print(input.min(), input.max())

        # added
        if self.training and self.observe_steps_remaining > 0:

            # added
            if self.usecosine:
                input = F.normalize(input, dim=-1)

            # if self.training:
            self.stream_cluster(input, expected_num_tokens)

            return (
                input,
                torch.zeros(input.shape[0],
                            dtype=torch.long, device=input.device),
                torch.tensor(0.0, dtype=input.dtype, device=input.device),
                None
            )

        dtype = input.dtype
        flatten = input.reshape(-1, self.dim)

        # if self.usecosine:
        #     # * H
        #     flatten = F.normalize(flatten, dim=-1)
        #     # H K
        #     embed = F.normalize(self.embed, dim=0)
            # * K
            # dist = flatten @ embed
        # else:

        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )

        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = F.embedding(embed_ind, self.embed.transpose(0, 1))

        if self.training:
            if self.ema:

                if self.usecosine:

                    # K
                    bins = embed_onehot.sum(0)
                    ema_inplace(self.cluster_size, bins, self.decay)
                    zero_mask = (bins == 0)
                    bins = bins.masked_fill(zero_mask, 1.)

                    # h * @ * K -> H K
                    embed_sum = flatten.transpose(0, 1) @ embed_onehot

                    # K H
                    embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
                    embed_normalized = F.normalize(embed_normalized, dim=1)
                    # K H
                    embed_normalized = torch.where(zero_mask[..., None], embed.t(),
                                                   embed_normalized)

                    embed_normalized = embed_normalized.t()
                    ema_inplace(self.embed, embed_normalized, self.decay)
                else:
                    cluster_size = embed_onehot.sum(0)

                    if torch.cuda.device_count() > 1 and distributed.get_world_size() > 1:
                        distributed.all_reduce(cluster_size)

                    ema_inplace(self.cluster_size, cluster_size, self.decay)
                    embed_sum = flatten.transpose(0, 1) @ embed_onehot

                    if torch.cuda.device_count() > 1 and distributed.get_world_size() > 1:
                        distributed.all_reduce(embed_sum)

                    ema_inplace(self.embed_avg, embed_sum, self.decay)
                    cluster_size = (
                        laplace_smoothing(self.cluster_size, self.n_embed, self.eps)
                        * self.cluster_size.sum()
                    )
                    embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
                    self.embed.data.copy_(embed_normalized)


        loss = F.mse_loss(input, quantize.detach()) * self.commitment

        quantize = input + (quantize - input).detach()
        return quantize, embed_ind, loss, dist
