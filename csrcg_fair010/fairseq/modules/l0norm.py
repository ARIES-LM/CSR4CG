# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Defines common utilities for l0-regularization layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import tensorflow as tf

import torch
import math

# Small constant value to add when taking logs or sqrts to avoid NaNs
EPSILON = 1e-8

# The default hard-concrete distribution parameters
BETA = 2.0 / 3.0
GAMMA = -0.1
ZETA = 1.1


def hard_concrete_sample(
        log_alpha,
        beta=BETA,
        gamma=GAMMA,
        zeta=ZETA,
        eps=EPSILON):
    """Sample values from the hard concrete distribution.

      The hard concrete distribution is described in
      https://arxiv.org/abs/1712.01312.

      Args:
        log_alpha: The log alpha parameters that control the "location" of the
          distribution.
        beta: The beta parameter, which controls the "temperature" of
          the distribution. Defaults to 2/3 from the above paper.
        gamma: The gamma parameter, which controls the lower bound of the
          stretched distribution. Defaults to -0.1 from the above paper.
        zeta: The zeta parameters, which controls the upper bound of the
          stretched distribution. Defaults to 1.1 from the above paper.
        eps: A small constant value to add to logs and sqrts to avoid NaNs.

      Returns:
        A tf.Tensor representing the output of the sampling operation.
    """
    # random_noise = tf.random_uniform(
    #     tf.shape(log_alpha),
    #     minval=0.0,
    #     maxval=1.0)

    random_noise = torch.ones_like(log_alpha).uniform_()

    # NOTE: We add a small constant value to the noise before taking the
    # log to avoid NaNs if a noise value is exactly zero. We sample values
    # in the range [0, 1), so the right log is not at risk of NaNs.
    gate_inputs = torch.log(random_noise + eps) - torch.log(1.0 - random_noise)
    gate_inputs = torch.sigmoid((gate_inputs + log_alpha) / beta)
    stretched_values = gate_inputs * (zeta - gamma) + gamma

    # return tf.clip_by_value(
    #     stretched_values,
    #     clip_value_max=1.0,
    #     clip_value_min=0.0)

    return torch.clamp(stretched_values, max=1.0, min=0.0)


def hard_concrete_mean(log_alpha, gamma=GAMMA, zeta=ZETA):
    """Calculate the mean of the hard concrete distribution.

      The hard concrete distribution is described in
      https://arxiv.org/abs/1712.01312.

      Args:
        log_alpha: The log alpha parameters that control the "location" of the
          distribution.
        gamma: The gamma parameter, which controls the lower bound of the
          stretched distribution. Defaults to -0.1 from the above paper.
        zeta: The zeta parameters, which controls the upper bound of the
          stretched distribution. Defaults to 1.1 from the above paper.

      Returns:
        A tf.Tensor representing the calculated means.
    """
    stretched_values = torch.sigmoid(log_alpha) * (zeta - gamma) + gamma
    return torch.clip(stretched_values, max=1.0, min=0.0)


def l0_norm(
        log_alpha,
        beta=BETA,
        gamma=GAMMA,
        zeta=ZETA):
    """Calculate the l0-regularization contribution to the loss.
      Args:
        log_alpha: Tensor of the log alpha parameters for the hard concrete
          distribution.
        beta: The beta parameter, which controls the "temperature" of
          the distribution. Defaults to 2/3 from the above paper.
        gamma: The gamma parameter, which controls the lower bound of the
          stretched distribution. Defaults to -0.1 from the above paper.
        zeta: The zeta parameters, which controls the upper bound of the
          stretched distribution. Defaults to 1.1 from the above paper.
      Returns:
        Scalar tensor containing the unweighted l0-regularization term contribution
        to the loss.
    """
    # Value of the CDF of the hard-concrete distribution evaluated at 0
    reg_per_weight = torch.sigmoid(log_alpha - beta * math.log(-gamma / zeta))
    return reg_per_weight


def var_train(
        weight_parameters,
        beta=BETA,
        gamma=GAMMA,
        zeta=ZETA,
        eps=EPSILON):
    """Model training, sampling hard concrete variables"""
    theta, log_alpha = weight_parameters

    # Sample the z values from the hard-concrete distribution
    weight_noise = hard_concrete_sample(
        log_alpha,
        beta,
        gamma,
        zeta,
        eps)
    weights = theta * weight_noise

    return weights, weight_noise


def l0_regularization_loss(l0_norm_loss,
                           step=1,
                           reg_scalar=1.0,
                           start_reg_ramp_up=0,
                           end_reg_ramp_up=1000,
                           warm_up=True):
    """Calculate the l0-norm weight for this iteration"""
    # step = tf.train.get_or_create_global_step()

    # current_step_reg = torch.max(
    #     0.0,
    #     tf.cast(step - start_reg_ramp_up, tf.float32))

    if warm_up:
        current_step_reg = max(0.0, step - start_reg_ramp_up)
        fraction_ramp_up_completed = min(current_step_reg / (end_reg_ramp_up - start_reg_ramp_up), 1.0)

        # regularizer intensifies over the course of ramp-up
        reg_scalar = fraction_ramp_up_completed * reg_scalar

    l0_norm_loss = reg_scalar * l0_norm_loss
    return l0_norm_loss


def var_eval(
        weight_parameters,
        gamma=GAMMA,
        zeta=ZETA):
    """Model evaluation, obtain mean value"""
    theta, log_alpha = weight_parameters

    # Use the mean of the learned hard-concrete distribution as the
    # deterministic weight noise at evaluation time
    weight_noise = hard_concrete_mean(log_alpha, gamma, zeta)
    weights = theta * weight_noise
    return weights, weight_noise


def extract_encodes(source_memory, source_mask, l0_mask, print_dprate=True):
    # x_shp =
    # B T
    l0_mask = l0_mask.bool().squeeze(-1) & source_mask
    l0_mask = l0_mask.to(source_memory.dtype)

    # count retained encodings
    k_value = torch.max(torch.sum(l0_mask.long(), 1)).long()
    # B K
    _, topk_indices = torch.topk(l0_mask, k_value)

    # prepare coordinate
    # x_pos = util.batch_coordinates(x_shp[0], k_value)
    batch_size = source_memory.size(0)
    # batch_pos = torch.arange(batch_size * k_value) // k_value
    # x_pos = batch_pos.reshape([batch_size, k_value])

    # B K 2
    # coord = torch.stack([x_pos, topk_indices], 2)

    # gather retained features
    # B K H
    # g_x = tf.gather_nd(source_memory, coord)
    # B K
    # g_mask = tf.gather_nd(l0_mask, coord)

    # B K H
    g_x = source_memory[torch.arange(batch_size).unsqueeze(1), topk_indices]
    g_mask = l0_mask[torch.arange(batch_size).unsqueeze(1), topk_indices]

    # padding zero
    # g_x = tf.pad(g_x, [[0, 0], [1, 0], [0, 0]])

    # pad = g_x.new_ones([batch_size, 1, g_x.size(-1)])
    # B 1+K H
    # g_x = torch.cat((pad, g_x), 1)

    # generate counts, i.e. how many tokens are dropped
    # B
    # droped_number = source_mask.sum(1) - l0_mask.sum(1)
    # pad_mask = (droped_number > 0.).float()
    # droped_number = torch.where(droped_number <= 0., torch.ones_like(droped_number), droped_number)
    # count_mask = torch.ones_like(g_mask)
    # count_mask = torch.cat([droped_number.unsqueeze(1), count_mask], 1)
    # B 1+K
    # g_mask = torch.cat([pad_mask.unsqueeze(1), g_mask], 1)

    return g_x, g_mask, None
