# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import math
from dataclasses import dataclass

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

import torch

@dataclass
class CrossEntropyCriterionConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")


@register_criterion("cross_entropy", dataclass=CrossEntropyCriterionConfig)
class CrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion("cross_entropyv2")
class CrossEntropyCriterionv2(
    CrossEntropyCriterion
):
    '''
    :return log_probs and log js-div
    '''
    def __init__(self, task, sentence_avg, jslamb, augnum, var, validvar, augnum4ce,):
        super().__init__(task, sentence_avg)
        self.jslamb = jslamb
        self.augnum = augnum
        self.var = var
        self.validvar = validvar
        self.augnum4ce = augnum4ce

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        CrossEntropyCriterion.add_args(parser)
        parser.add_argument("--jslamb", default=0.0, type=float, metavar="D",)
        parser.add_argument("--augnum", default=1, type=int)
        parser.add_argument("--var", type=str, choices=['logp', 'loss', 'p', 'rd', 'none'])
        parser.add_argument("--validvar", type=int, default=0, help='used for inference, means the augnum')
        parser.add_argument("--augnum4ce", default=1, type=int)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)
        loss = F.nll_loss(
            lprobs,
            target,
            ignore_index=self.padding_idx,
            reduction="sum" if reduce else "none",
        )
        return loss, loss, lprobs

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        # added
        if self.training or self.validvar:
            augnum = self.augnum
            oribsz = sample['target'].size(0)
            # BT 1
            target_mask = sample['target'].eq(self.padding_idx).view(-1, 1)

            if augnum > 1:
                sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].repeat(augnum, 1)
                sample['net_input']['src_lengths'] = sample['net_input']['src_lengths'].repeat(augnum)
                sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].repeat(augnum, 1)
                # sample['ntokens'] *= 2
                sample['target'] = sample['target'].repeat(augnum, 1)

            # open dp
            model.train()

            net_output = model(**sample["net_input"])

            # KB T V
            logits, extra = net_output
            _, seqlen, vocabsize = logits.size()

            '''
            if self.augnum4ce < augnum:
                # K B T V
                logits = logits.view(augnum, -1, logits.size(-1))

                lprobs = F.log_softmax(logits, dim=-1)

                augnum4ce = self.augnum4ce

                logits4ce = logits[:augnum4ce].view(-1, seqlen, vocabsize)
                sample4ce = {'target': sample['target'][:oribsz*augnum4ce]}

                loss, nll_loss, _ = self.compute_loss(model, (logits4ce, None), sample4ce, reduce=False)
                loss = loss.sum() / augnum4ce
            else:
            '''

            loss, nll_loss, lprobs = self.compute_loss(model, net_output, sample, reduce=False)
            loss = loss.sum() / self.augnum4ce

            classvarloss = extra.get('varloss', torch.zeros_like(loss)) * sample_size

            if self.var == 'loss':
                # K B T
                loss = loss.view(augnum, -1, sample['target'].size(-1))
                var = torch.var(loss, dim=0, unbiased=True)
            elif self.var == 'logp':
                lprobs = lprobs.view(augnum, -1, lprobs.size(-1))
                var = torch.var(lprobs, dim=0, unbiased=True)
                # BT V
                var.masked_fill_(target_mask, 0.)
            elif self.var == 'p':
                lprobs = lprobs.exp().view(augnum, -1, lprobs.size(-1))
                var = torch.var(lprobs, dim=0, unbiased=True)
                # BT V
                var.masked_fill_(target_mask, 0.)
            elif self.var == 'rd':
                # 2BT V
                lprobs1, lprobs2 = lprobs.split(lprobs.size(0)//2, dim=0)
                # BT V
                klloss = torch.nn.functional.kl_div(lprobs2, lprobs1, reduction='none', log_target=True)
                klloss2 = torch.nn.functional.kl_div(lprobs1, lprobs2, reduction='none', log_target=True)
                klloss.masked_fill_(target_mask, 0.)
                klloss2.masked_fill_(target_mask, 0.)
                var = (klloss + klloss2) / 2.0
            else:
                var = torch.zeros_like(loss)

            varloss = var.sum() * self.jslamb
            loss = loss + varloss

            loss = loss + classvarloss

            logging_output = {"loss": loss.data, "ntokens": sample["ntokens"],
                              "nsentences": sample["target"].size(0), "sample_size": sample_size,
                              'jsloss': varloss.data, 'varloss': classvarloss.data}

        else:
            net_output = model(**sample["net_input"])
            loss, nll_loss, lprobs = self.compute_loss(model, net_output, sample, reduce=reduce)

            logging_output = {
                "loss": loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        jsloss_sum = sum(log.get("jsloss", 0) for log in logging_outputs)
        varloss_sum = sum(log.get("varloss", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        metrics.log_scalar(
            "jsloss", jsloss_sum / sample_size / math.log(2), sample_size, round=4
        )
        metrics.log_scalar(
            "varloss", varloss_sum / sample_size / math.log(2), sample_size, round=4
        )

        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )


