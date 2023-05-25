# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
from torch import nn
from torch.nn import functional as F


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion(
    "label_smoothed_cross_entropy", dataclass=LabelSmoothedCrossEntropyCriterionConfig
)
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )

        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }

        # added
        # logits, _ = net_output
        # with torch.no_grad():
        #     logitsnorm = logits.norm(dim=-1).mean()
        #     logging_output['logitsnorm'] = logitsnorm.data

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):

        lprobs = model.get_normalized_probs(net_output, log_probs=True)

        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        # added
        logitnorm_sum = sum(log.get("logitsnorm", 0) for log in logging_outputs)
        metrics.log_scalar(
            "logitnorm", logitnorm_sum / math.log(2), sample_size, round=3
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


def js_kiv(lprobs1, lprobs2):
    klloss1 = F.kl_div(lprobs1, lprobs2, reduction='none', log_target=True)
    klloss2 = F.kl_div(lprobs2, lprobs1, reduction='none', log_target=True)
    return (klloss2 + klloss1) / 2


@register_criterion("label_smoothed_cross_entropy_set")
class SetLabelSmoothedCrossEntropyCriterion(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size,
            report_accuracy,
            jslamb, augnum, var, validvar, augnum4ce,
    ):
        super().__init__(
            task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy
        )
        self.jslamb = jslamb
        self.augnum = augnum
        self.var = var
        self.validvar = validvar
        self.augnum4ce = augnum4ce

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument("--jslamb", default=0.0, type=float, )
        parser.add_argument("--augnum", default=1, type=int, )
        parser.add_argument("--var", type=str, choices=['logp', 'logits', 'logpsqrt', 'loss', 'p', 'none', 'js'])
        parser.add_argument("--validvar", type=int, default=0)
        parser.add_argument("--augnum4ce", type=int, default=1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss, lprobs

    def forward(self, model, sample, reduce=True):
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        # added
        if self.training:
            augnum = self.augnum
            oribsz = sample['target'].size(0)
            # BT 1
            target_mask = sample['target'].eq(self.padding_idx).view(-1, 1)

            if augnum > 1:
                sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].repeat(augnum, 1)
                sample['net_input']['src_lengths'] = sample['net_input']['src_lengths'].repeat(augnum)
                sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].repeat(augnum, 1)
                # sample['ntokens'] *= 2
                # BK T
                sample['target'] = sample['target'].repeat(augnum, 1)

            # added
            model.train()

            net_output = model(**sample["net_input"])
            # KB T V
            logits, extra = net_output
            _, seqlen, vocabsize = logits.size()

            loss, nll_loss, lprobs = self.compute_loss(model, net_output, sample, reduce=reduce)

            loss = loss
            nll_loss = nll_loss

            classvarloss = extra.get('varloss', torch.zeros_like(loss)) * sample_size

            if self.var == 'loss':
                # K B T
                loss = loss.view(augnum, -1, sample['target'].size(-1))
                var = torch.var(loss, dim=0, unbiased=True)
            elif self.var == 'logits':
                var = torch.var(logits, dim=0, unbiased=True)
                var = var.masked_fill(target_mask, 0.)
            elif self.var == 'logp':
                lprobs = lprobs.view(augnum, -1, lprobs.size(-1))
                var = torch.var(lprobs, dim=0, unbiased=True)
                # BT V
                var = var.masked_fill(target_mask, 0.)
            elif self.var == 'p':
                lprobs = lprobs.exp().view(augnum, -1, lprobs.size(-1))
                var = torch.var(lprobs, dim=0, unbiased=True)
                # BT V
                var.masked_fill_(target_mask, 0.)
            elif self.var == 'js':
                # 2BT V
                if self.augnum > 2:
                    lprobs = lprobs.view(augnum, -1, lprobs.size(-1))
                    #  pairwise
                    var = []
                    for i in range(augnum):
                        for j in range(i + 1, augnum):
                            var.append(js_kiv(lprobs[i], lprobs[j]))
                    var = torch.stack(var, 0).mean(0)
                    var.masked_fill_(target_mask, 0.)
                else:
                    lprobs1, lprobs2 = lprobs.split(lprobs.size(0) // 2, dim=0)
                    klloss = torch.nn.functional.kl_div(lprobs2, lprobs1, reduction='none', log_target=True)
                    klloss2 = torch.nn.functional.kl_div(lprobs1, lprobs2, reduction='none', log_target=True)
                    klloss.masked_fill_(target_mask, 0.)
                    klloss2.masked_fill_(target_mask, 0.)
                    var = torch.sum(klloss + klloss2) / 2.0
            else:
                var = torch.zeros_like(loss)

            varloss = var.sum()
            varloss = varloss * self.jslamb
            loss = loss + varloss

            loss = loss + classvarloss

            logging_output = {"loss": loss.data, "nll_loss": nll_loss.data, "ntokens": sample["ntokens"],
                              "nsentences": sample["target"].size(0), "sample_size": sample_size,
                              'jsloss': varloss.data, 'varloss': classvarloss.data}

        else:
            net_output = model(**sample["net_input"])
            loss, nll_loss, lprobs = self.compute_loss(model, net_output, sample, reduce=reduce)

            logging_output = {
                "loss": loss.data,
                "nll_loss": nll_loss.data,
                "ntokens": sample["ntokens"],
                "nsentences": sample["target"].size(0),
                "sample_size": sample_size,
            }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)

        # added
        jsloss_sum = sum(log.get("jsloss", 0) for log in logging_outputs)
        varloss_sum = sum(log.get("varloss", 0) for log in logging_outputs)

        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )

        # added
        metrics.log_scalar(
            "jsloss", jsloss_sum / sample_size / math.log(2), sample_size, round=4
        )
        metrics.log_scalar(
            "varloss", varloss_sum / sample_size / math.log(2), sample_size, round=4
        )

        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )




