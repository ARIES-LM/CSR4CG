# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import itertools
import json
import logging
import os
from statistics import mode

from typing import Optional
from argparse import Namespace
from omegaconf import II
import torch

import numpy as np
from fairseq import metrics, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
from fairseq.data.indexed_dataset import get_available_dataset_impl
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.optim.amp_optimizer import AMPOptimizer


EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,
    sorted_global=True,
    **kwargs
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("[{}]".format(src))
        )
        if tgt_dataset is not None:
            tgt_dataset = AppendTokenDataset(
                tgt_dataset, tgt_dict.index("[{}]".format(tgt))
            )
        eos = tgt_dict.index("[{}]".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
        sorted_global=sorted_global, # added
        **kwargs
    )


@dataclass
class TranslationConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"},)
    source_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "source language",
            "argparse_alias": "-s",
        },)
    target_lang: Optional[str] = field(
        default=None,
        metadata={
            "help": "target language",
            "argparse_alias": "-t",
        },)
    load_alignments: bool = field(
        default=False, metadata={"help": "load the binarized alignments"})
    left_pad_source: bool = field(
        default=True, metadata={"help": "pad the source on the left"})
    left_pad_target: bool = field(
        default=False, metadata={"help": "pad the target on the left"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    upsample_primary: int = field(
        default=-1, metadata={"help": "the amount of upsample primary dataset"}
    )
    truncate_source: bool = field(
        default=False, metadata={"help": "truncate source to max-source-positions"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={
            "help": "if >0, then bucket source and target lengths into "
            "N buckets and pad accordingly; this is useful on TPUs to minimize the number of compilations"
        },
    )
    train_subset: str = II("dataset.train_subset")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")

    # options for reporting BLEU during validation
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_bleu_args: Optional[str] = field(
        default="{}",
        metadata={
            "help": 'generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_bleu_detok: str = field(
        default="space",
        metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses'); required if using --eval-bleu; "
            "use 'space' to disable detokenization; see fairseq.data.encoders for other options"
        },
    )
    eval_bleu_detok_args: Optional[str] = field(
        default="{}",
        metadata={"help": "args for building the tokenizer, if needed, as JSON string"},
    )
    eval_tokenized_bleu: bool = field(
        default=False, metadata={"help": "compute tokenized BLEU instead of sacrebleu"}
    )
    eval_bleu_remove_bpe: Optional[str] = field(
        default=None,
        metadata={
            "help": "remove BPE before computing BLEU",
            "argparse_const": "@@ ",
        },
    )
    eval_bleu_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )

    # added
    mixfrom: int = field(
        default=0,
        metadata={
            "help": "if >0, then mixup from epoch $$"
        },
    )
    # added
    queue_epoch: int = field(default=0,
        metadata={
            "help": "for swav"
        },
    )
    # jslamb: float = field(default=0.0)
    dropnum: int = field(default=2)

    pretrain: bool = field(
        default=True, metadata={"help": "pretrain the base model without mix"}
    )

    close_shuffle: bool = field(
        default=False, metadata={"help": "shuffle all data"}
    )
    close_sort: bool = field(
        default=False, metadata={"help": "sort data by length"}
    )
    group_shuffle: int = field(default=0)
    token_sentidx_file: Optional[str] = field(default=None)


@register_task("translation", dataclass=TranslationConfig)
class TranslationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.
    """

    cfg: TranslationConfig

    def __init__(self, cfg: TranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, cfg: TranslationConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0
        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(paths[0])
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "dict.{}.txt".format(cfg.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))

        return cls(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            num_buckets = self.cfg.num_batch_buckets,
            shuffle=not self.cfg.close_shuffle and (split != "test"), # added
            pad_to_multiple=self.cfg.required_seq_len_multiple,
            sorted_global=not self.cfg.close_sort, # added
            group_shuffle=self.cfg.group_shuffle if split == "train" else 0, # added
            token_sentidx_file = self.cfg.token_sentidx_file if split == "train" else None
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

    # added
    def begin_epoch(self, epoch, model):
        """Hook function called before the start of each epoch."""
        # print(epoch) from 1
        pass

    def build_model(self, cfg):
        model = super().build_model(cfg)
        if self.cfg.eval_bleu:
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(self.cfg.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    # def train_step(
    #     self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    # ):
    #     """
    #     Do forward and backward, and return the loss as computed by *criterion*
    #     for the given *model* and *sample*.
    #
    #     Args:
    #         sample (dict): the mini-batch. The format is defined by the
    #             :class:`~fairseq.data.FairseqDataset`.
    #         model (~fairseq.models.BaseFairseqModel): the model
    #         criterion (~fairseq.criterions.FairseqCriterion): the criterion
    #         optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
    #         update_num (int): the current update
    #         ignore_grad (bool): multiply loss by 0 if this is set to True
    #
    #     Returns:
    #         tuple:
    #             - the loss
    #             - the sample size, which is used as the denominator for the
    #               gradient
    #             - logging outputs to display while training
    #     """
    #     model.train()
    #     model.set_num_updates(update_num)
    #     with torch.autograd.profiler.record_function("forward"):
    #         with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
    #             loss, sample_size, logging_output = criterion(model, sample)
    #
    #     if ignore_grad:
    #         loss *= 0
    #     with torch.autograd.profiler.record_function("backward"):
    #         optimizer.backward(loss)
    #     return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.cfg.eval_bleu:

            def sum_logs(key):
                import torch
                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.cfg.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.cfg.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])


@register_task("translation_rep", dataclass=TranslationConfig)
class TranslationRepTask(TranslationTask):
    def __init__(self, cfg: TranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    # added
    def begin_epoch(self, epoch, model):
        """Hook function called before the start of each epoch."""
        # print(epoch) from 1
        if self.cfg.mixfrom == epoch:
            model.start_mix()
            logger.info('from epoch {} enc mix {} dec mix {}'.format(epoch, model.encoder.mix, model.decoder.mix))


@register_task("translation_set", dataclass=TranslationConfig)
class TranslationSetTask(TranslationTask):
    def __init__(self, cfg: TranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

        # added
        self.jslamb = cfg.jslamb
        self.ampfrom = cfg.mixfrom

    # added
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        """

        model.train()
        model.set_num_updates(update_num)

        if update_num >= self.ampfrom:
            model.set_batchmp(True)
            with torch.autograd.profiler.record_function("forward"):
                with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                    loss1, _, _, nll_loss1, lprobs1 = criterion(model, sample)
                    # loss1, _, _ = criterion(model, sample)
        else:
            loss1 = 0

        model.set_batchmp(False)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss2, sample_size, logging_output, _, lprobs2 = criterion(model, sample)
                # loss2, sample_size, logging_output = criterion(model, sample)

        # loss
        # print(loss1, loss2)
        if update_num >= self.ampfrom:
            loss = (loss2 + loss1) / 2

            if self.jslamb > 0:
                # token-num V
                klloss = torch.nn.functional.kl_div(lprobs2, lprobs1, reduction='none', log_target=True)
                klloss2 = torch.nn.functional.kl_div(lprobs1, lprobs2, reduction='none', log_target=True)

                target_mask = sample['target'] == criterion.padding_idx

                klloss = (klloss + klloss2) / 2
                klloss.masked_fill_(target_mask.view(-1, 1), 0)

                klloss = self.jslamb * klloss.sum()
                # print(klloss, loss1, loss2)
                logging_output['jsloss'] = klloss.data
                loss = loss + klloss
        else:
            loss = loss2

        # logging_output['total_loss'] = loss.data

        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output


@register_task("translation_rdrop", dataclass=TranslationConfig)
class TranslationRDropTask(TranslationTask):
    def __init__(self, cfg: TranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

        # added
        # self.jslamb = cfg.jslamb
        self.mixfrom = cfg.mixfrom

        self.dropnum = cfg.dropnum

    # added
    def begin_epoch(self, epoch, model):
        """Hook function called before the start of each epoch."""
        # print(epoch) from 1
        if self.cfg.mixfrom > 0 and epoch >= self.cfg.mixfrom:
            model.start_mix()
            logger.info('from epoch {} enc mix {}'.format(epoch, model.encoder.mix))

    # added
    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        """

        model.train()
        model.set_num_updates(update_num)

        oribz = sample['target'].size(0)

        augnum = self.dropnum

        # sample['net_input']['src_tokens'] = sample['net_input']['src_tokens'].repeat(augnum, 1)
        # sample['net_input']['src_lengths'] = sample['net_input']['src_lengths'].repeat(augnum)
        # sample['net_input']['prev_output_tokens'] = sample['net_input']['prev_output_tokens'].repeat(augnum, 1)
        # # BT 1
        # target_mask = sample['target'].eq(criterion.padding_idx).view(-1, 1)
        # sample['target'] = sample['target'].repeat(augnum, 1)

        # print(sample['target'][0])
        # print(sample['target'][oribz])
        # exit()

        # with torch.autograd.profiler.record_function("forward"):
        #     with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
        #         loss1, _, _, nll_loss1, lprobs1 = criterion(model, sample)

        # with torch.autograd.profiler.record_function("forward"):
        #     with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
        #         loss2, sample_size, logging_output, _, lprobs2 = criterion(model, sample)

        # if update_num >= self.mixfrom:
        #     model.start_mix()

        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output, _, lprobs = criterion(model, sample)

        # variance
        # if augnum > 2:
        #     probs = torch.exp(lprobs)
        #     probs = probs.view(augnum, -1, probs.size(-1))
        #     var = torch.var(probs, dim=0, unbiased=True)
        #     # BT V
        #     var.masked_fill_(target_mask, 0.)
        #     varloss = var.sum() / augnum
        #     varloss = varloss * self.jslamb
        #     # print(varloss)
        #     # exit()
        #     # print(loss)
        #     loss = loss / augnum
        #     # print(loss)
        #     # exit()
        #
        #     loss = loss + varloss
        #     logging_output['jsloss'] = varloss.data
        # elif augnum == 2:
        #     # print('sample_size', sample_size)
        #     # loss = (loss1 + loss2) / 2
        #
        #     # 2BT V
        #     lprobs1, lprobs2 = lprobs.split(lprobs.size(0)//2, dim=0)
        #
        #     klloss = torch.nn.functional.kl_div(lprobs2, lprobs1, reduction='none', log_target=True)
        #     klloss2 = torch.nn.functional.kl_div(lprobs1, lprobs2, reduction='none', log_target=True)
        #
        #     klloss.masked_fill_(target_mask, 0.)
        #     klloss2.masked_fill_(target_mask, 0.)
        #
        #     klloss = klloss.sum()
        #     klloss2 = klloss2.sum()
        #     klloss = (klloss + klloss2) / 2
        #
        #     klloss = self.jslamb * klloss
        #     logging_output['jsloss'] = klloss.data
        #
        #     loss = loss + klloss
        #
        # logging_output['loss'] = loss.data

        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output


@register_task("translation_cl", dataclass=TranslationConfig)
class TranslationCLTask(TranslationTask):
    def __init__(self, cfg: TranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

        # added
        self.jslamb = cfg.jslamb
        self.mixfrom = cfg.mixfrom

    # added
    def begin_epoch(self, epoch, model):
        """Hook function called before the start of each epoch."""
        # print(epoch) from 1
        if self.cfg.mixfrom > 0 and epoch >= self.cfg.mixfrom:
            model.start_mix()
            logger.info('from epoch {} enc mix {}'.format(epoch, model.encoder.mix))


@register_task("translation_swav", dataclass=TranslationConfig)
class TranslationSwavTask(TranslationTask):
    def __init__(self, cfg: TranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    # added
    def begin_epoch(self, epoch, model):
        """Hook function called before the start of each epoch."""
        # print(epoch) from 1

        # if model.__name__ == 'TransformerRepModel':
        if self.cfg.queue_epoch == epoch:
            model.build_swav_queue()
            logger.info('from epoch {} swav start queue'.format(epoch))

        if model.w2cfile is not None:
            logger.info('print clusters of each words')

            with open(model.w2cfile, 'w') as f:
                for w in model.word_clusters:
                    w_qs = set(model.word_clusters[w])
                    print(w, len(w_qs), '|', *w_qs, file=f)

            model.word_clusters.clear()


@register_task("translation_vq", dataclass=TranslationConfig)
class TranslationVQTask(TranslationTask):
    def __init__(self, cfg: TranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    # added
    def begin_epoch(self, epoch, model):
        """Hook function called before the start of each epoch."""
        # print(epoch) from 1

        codeuse = model.encoder.codeuse
        
        print(codeuse[:50])

        # logger.info('code useage sum: {}'.format(codeuse.sum().item()))

        usedcode = (codeuse > 0).sum().float() / len(codeuse)
        logger.info('code frequency, max:{}, useage: {}'.format(codeuse.max().item(),
                                                                         usedcode.item()))

        # reset
        model.encoder.codeuse.zero_()


# def load_langpair_datasetx(
#     data_path,
#     split,
#     src,
#     src_dict,
#     tgt,
#     tgt_dict,
#     combine,
#     dataset_impl,
#     upsample_primary,
#     left_pad_source,
#     left_pad_target,
#     max_source_positions,
#     max_target_positions,
#     prepend_bos=False,
#     load_alignments=False,
#     truncate_source=False,
#     append_source_id=False,
#     num_buckets=0,
#     shuffle=True,
#     pad_to_multiple=1,
#     prepend_bos_src=None,
# ):
#     def split_exists(split, src, tgt, lang, data_path):
#         filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
#         return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

#     src_datasets = []
#     tgt_datasets = []

#     for k in itertools.count():
#         split_k = split + (str(k) if k > 0 else "")

#         # infer langcode
#         if split_exists(split_k, src, tgt, src, data_path):
#             prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
#         elif split_exists(split_k, tgt, src, src, data_path):
#             prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
#         else:
#             if k > 0:
#                 break
#             else:
#                 raise FileNotFoundError(
#                     "Dataset not found: {} ({})".format(split, data_path)
#                 )

#         src_dataset = data_utils.load_indexed_dataset(
#             prefix + src, src_dict, dataset_impl
#         )
#         if truncate_source:
#             src_dataset = AppendTokenDataset(
#                 TruncateDataset(
#                     StripTokenDataset(src_dataset, src_dict.eos()),
#                     max_source_positions - 1,
#                 ),
#                 src_dict.eos(),
#             )
#         src_datasets.append(src_dataset)

#         tgt_dataset = data_utils.load_indexed_dataset(
#             prefix + tgt, tgt_dict, dataset_impl
#         )
#         if tgt_dataset is not None:
#             tgt_datasets.append(tgt_dataset)

#         logger.info(
#             "{} {} {}-{} {} examples".format(
#                 data_path, split_k, src, tgt, len(src_datasets[-1])
#             )
#         )

#         if not combine:
#             break

#     assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

#     if len(src_datasets) == 1:
#         src_dataset = src_datasets[0]
#         tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
#     else:
#         sample_ratios = [1] * len(src_datasets)
#         sample_ratios[0] = upsample_primary
#         src_dataset = ConcatDataset(src_datasets, sample_ratios)
#         if len(tgt_datasets) > 0:
#             tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
#         else:
#             tgt_dataset = None

#     if prepend_bos:
#         assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
#         src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
#         if tgt_dataset is not None:
#             tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
#     elif prepend_bos_src is not None:
#         logger.info(f"prepending src bos: {prepend_bos_src}")
#         src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

#     eos = None
#     if append_source_id:
#         src_dataset = AppendTokenDataset(
#             src_dataset, src_dict.index("[{}]".format(src))
#         )
#         if tgt_dataset is not None:
#             tgt_dataset = AppendTokenDataset(
#                 tgt_dataset, tgt_dict.index("[{}]".format(tgt))
#             )
#         eos = tgt_dict.index("[{}]".format(tgt))

#     align_dataset = None
#     if load_alignments:
#         align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
#         if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
#             align_dataset = data_utils.load_indexed_dataset(
#                 align_path, None, dataset_impl
#             )

#     tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None

#     return LanguagePairDatasetX(
#         src_dataset,
#         src_dataset.sizes,
#         src_dict,
#         tgt_dataset,
#         tgt_dataset_sizes,
#         tgt_dict,
#         left_pad_source=left_pad_source,
#         left_pad_target=left_pad_target,
#         align_dataset=align_dataset,
#         eos=eos,
#         num_buckets=num_buckets,
#         shuffle=shuffle,
#         pad_to_multiple=pad_to_multiple,
#     )


@register_task("translation_vqx", dataclass=TranslationConfig)
class TranslationVQXTask(TranslationVQTask):
    def __init__(self, cfg: TranslationConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    def begin_epoch(self, epoch, model):
        """Hook function called before the start of each epoch."""
        # print(epoch) from 1

        codeuse = model.codeuse
        
        print(codeuse[:50])

        # logger.info('code useage sum: {}'.format(codeuse.sum().item()))

        usedcode = (codeuse > 0).sum() / len(codeuse)
        logger.info('code useage: {}'.format(usedcode.item()))
        
        if getattr(self, "codeuse", None) is not None:
            prev_codeuse_p = self.codeuse / self.codeuse.sum()
            codeuse_p = codeuse/codeuse.sum()

            mean = ((prev_codeuse_p+codeuse_p)/2).log()
            kl = torch.nn.KLDivLoss(log_target=True)

            js = 0.5*(kl(mean, codeuse_p.log()) + kl(mean, prev_codeuse_p.log()))

            logger.info('code js: {}'.format(js.item()))
        else:
             self.codeuse = codeuse.clone().detach()

        # reset
        model.codeuse -= codeuse

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample)

        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)

        # print(model.input_quantizer.embedding.grad)
        # print(model.encfc1.weight.grad)
        #
        # exit()

        return loss, sample_size, logging_output


    # def load_dataset(self, split, epoch=1, combine=False, **kwargs):
    #     """Load a given dataset split.

    #     Args:
    #         split (str): name of the split (e.g., train, valid, test)
    #     """
    #     paths = utils.split_paths(self.cfg.data)
    #     assert len(paths) > 0
    #     if split != self.cfg.train_subset:
    #         # if not training data set, use the first shard for valid and test
    #         paths = paths[:1]
    #     data_path = paths[(epoch - 1) % len(paths)]

    #     # infer langcode
    #     src, tgt = self.cfg.source_lang, self.cfg.target_lang

    #     self.datasets[split] = load_langpair_datasetx(
    #         data_path,
    #         split,
    #         src,
    #         self.src_dict,
    #         tgt,
    #         self.tgt_dict,
    #         combine=combine,
    #         dataset_impl=self.cfg.dataset_impl,
    #         upsample_primary=self.cfg.upsample_primary,
    #         left_pad_source=self.cfg.left_pad_source,
    #         left_pad_target=self.cfg.left_pad_target,
    #         max_source_positions=self.cfg.max_source_positions,
    #         max_target_positions=self.cfg.max_target_positions,
    #         load_alignments=self.cfg.load_alignments,
    #         truncate_source=self.cfg.truncate_source,
    #         num_buckets=self.cfg.num_batch_buckets,
    #         shuffle=(split != "test"),
    #         pad_to_multiple=self.cfg.required_seq_len_multiple,
    #     )


#     def train_step(
#         self, sample, model, criterion, optimizer, update_num, ignore_grad=False
#     ):
#         """
#         Do forward and backward, and return the loss as computed by *criterion*
#         for the given *model* and *sample*.
    
#         Args:
#             sample (dict): the mini-batch. The format is defined by the
#                 :class:`~fairseq.data.FairseqDataset`.
#             model (~fairseq.models.BaseFairseqModel): the model
#             criterion (~fairseq.criterions.FairseqCriterion): the criterion
#             optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
#             update_num (int): the current update
#             ignore_grad (bool): multiply loss by 0 if this is set to True
    
#         Returns:
#             tuple:
#                 - the loss
#                 - the sample size, which is used as the denominator for the
#                   gradient
#                 - logging outputs to display while training
#         """
#         model.train()
#         model.set_num_updates(update_num)
#         with torch.autograd.profiler.record_function("forward"):
#             with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
#                 loss, sample_size, logging_output = criterion(model, sample)

#                 # added
                

#                 loss2, _, _ = criterion(model, sample)

    
#         if ignore_grad:
#             loss *= 0
#         with torch.autograd.profiler.record_function("backward"):
#             optimizer.backward(loss)
#         return loss, sample_size, logging_output
