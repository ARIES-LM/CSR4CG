#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""

import argparse
import gc
import logging
import math
import os
import sys
from typing import Dict, Optional, Any, List, Tuple, Callable

# We need to setup root logger before importing any fairseq libraries.
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.train")

import numpy as np
import torch
from fairseq import (
    checkpoint_utils,
    options,
    quantization_utils,
    tasks,
    utils,
)
from fairseq.data import iterators, data_utils
from fairseq.data.plasma_utils import PlasmaStore
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap, utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics, progress_bar
from fairseq.model_parallel.megatron_trainer import MegatronTrainer
from fairseq.trainer import Trainer
from omegaconf import DictConfig, OmegaConf

import time
from string import punctuation
from sklearn.cluster import MiniBatchKMeans, KMeans


def cluster_dict(cfg, trainer, task, epoch_itr):
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=False,
        set_dataset_epoch=False,
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    # itr = iterators.GroupedIterator(itr, update_freq)

    # trainer.begin_epoch(epoch_itr.epoch)

    should_stop = False

    # clustering parameters
    # select_layer = cfg.common.clus_layer
    n_cluster = cfg.common.n_cluster
    partial_fit = False
    onehead = False

    clustask = 'mt-src'
    # clustask = 'lm'

    # 0: embedding only
    feature_type = 1

    minibatch = 1024 * 20
    seed = cfg.common.seed
    # kmeans_model = MiniBatchKMeans(n_clusters=n_cluster, random_state=seed, batch_size=minibatch)
    kmeans_model = KMeans(n_clusters=n_cluster, random_state=seed)

    # for partial fit
    kmeans_epoch = 10

    # paths = utils.split_paths(args.data)
    # source_lang, target_lang = data_utils.infer_language_pair(paths[0])
    # src_dict = os.path.join(paths[0], "dict.txt".format(source_lang))
    # tgt_dict = os.path.join(paths[0], "dict.{}.txt".format(target_lang))

    def dict_freq(dict_path):
        # w_freq = {}
        w_freq = []
        for l in open(src_dict):
            w, freq = l.strip().split()

            if w in set(punctuation):
                continue
            # w_freq.append([w, freq])
            w_freq.append(task.dictionary.index(w))
            # w_freq[w] = int(freq)
        return w_freq

    # src_dict_freq = dict_freq(src_dict)
    path = cfg.checkpoint.save_dir

    t1 = time.time()

    # np.set_printoptions(threshold=sys.maxsize)

    if feature_type == 0:
        # rm special tokens
        embeddings = trainer.model.encoder.embed_tokens[task.src_dict.n_special:]
        embeddings = embeddings.weight.cpu().detach().numpy()

        # N H
        kmeans_model.fit(embeddings)
        labels = kmeans_model.labels_

        for c in range(n_cluster):
            c_wids = np.where(labels == c)[0]
            c_w = [task.src_dict[wid] for wid in c_wids]
            print(c, len(c_w), *c_w)

        exit()

    if 'mt' in clustask:

        useless_indx = []
        clusnum_perline = []

        if clustask == 'mt-src':
            exwords = [trainer.task.src_dict.eos(), trainer.task.src_dict.unk(), trainer.task.src_dict.pad()]
            # for punc in punctuation:
            #     if punc in trainer.task.src_dict:
            #         exwords.append(trainer.task.src_dict.index(punc))
        else:
            exwords = [trainer.task.tgt_dict.eos(), trainer.task.tgt_dict.unk(), trainer.task.tgt_dict.pad()]
            # for punc in punctuation:
            #     if punc in trainer.task.tgt_dict:
            #         exwords.append(trainer.task.tgt_dict.index(punc))

        print('exwords')
        print(exwords)
        exwords = set(exwords)

    trainer.model.eval()

    if not partial_fit:
        all_hiddens = []

        itr = epoch_itr.next_epoch_itr(
            fix_batches_to_gpus=False,
            shuffle=False,
        )

        with torch.no_grad():
            for i_s, sample in enumerate(itr):  # delayed update loop
                sample, _ = trainer._prepare_sample(sample)

                # no bos and eos
                if clustask == 'mt-src':
                    encoder_out = trainer.model.encoder(sample['net_input']['src_tokens'],
                                                        return_all_hiddens=True)

                    states = encoder_out['encoder_states']
                    # T B H
                    ctxstate = states[-1]

                    # sentence_rep
                    # B T
                    ctxstate = ctxstate.transpose(0,1)
                    mask = encoder_out['encoder_padding_mask'][0]
                    ctxstate.masked_fill_(mask.unsqueeze(-1), 0)
                    # B H
                    state = ctxstate.sum(1) / sample['net_input']['src_lengths']

                    # head_num = cfg.model.decoder_attention_heads
                    # ctxstate = ctxstate.reshape(ctxstate.size(0), ctxstate.size(1), head_num,
                    #                             ctxstate.size(-1) // head_num)
                    # ctxstate = ctxstate.mean(2)
                    #
                    # state = torch.cat((states[0], ctxstate), -1)
                    # state = state.transpose(0, 1)

                elif clustask == 'mt-tgt':
                    state, extra = trainer.model.lm_forward(**sample['net_input'],
                                                            features_only=True,
                                                            return_all_hiddens=True)
                elif clustask == 'lm':
                    state, extra = trainer.model(sample['net_input']['src_tokens'],
                                                 features_only=True,
                                                 return_all_hiddens=True)

                # if onehead:
                #     head_num = args.decoder_attention_heads
                #     state = state.reshape(state.size(0), state.size(1), head_num, state.size(-1) // head_num)
                #     state = state.mean(2)
                #
                # if 'mt' in clustask:
                #     if clustask == 'mt-src':
                #         get_sample = sample['net_input']['src_tokens']
                #     else:
                #         get_sample = sample['target']
                #
                #     for ib, l in enumerate(get_sample):
                #         # eos, punc
                #         # print(l)
                #         useless_indx_i = []
                #         clusnum_perline_i = 0
                #
                #         for iw, w in enumerate(l):
                #             if w.item() not in exwords:
                #                 all_hiddens.append(state[ib, iw].cpu().numpy())
                #                 clusnum_perline_i += 1
                #             else:
                #                 useless_indx_i.append(iw)
                #
                #         clusnum_perline.append(clusnum_perline_i)
                #         useless_indx.append(useless_indx_i)
                # else:
                #     for ib, l in enumerate(sample['net_input']['src_lengths']):
                #         state_i = state[ib, :l].cpu().numpy()
                #         all_hiddens.extend([si for si in state_i])

                all_hiddens.append(state.cpu().numpy())

        # all_hiddens = np.stack(all_hiddens)
        all_hiddens = np.concatenate(all_hiddens, 0)
        print('all hidden shape', all_hiddens.shape)
        kmeans_model.fit(all_hiddens)

    else:
        # todo
        with torch.no_grad():
            for epc in range(kmeans_epoch):

                all_hiddens = []

                itr = epoch_itr.next_epoch_itr(
                    fix_batches_to_gpus=args.fix_batches_to_gpus,
                    shuffle=False,
                )

                for _, sample in enumerate(itr):  # delayed update loop

                    sample = trainer._prepare_sample(sample)
                    # if sample is None:
                    #     # when sample is None, run forward/backward on a dummy batch
                    #     # and ignore the resulting gradients
                    #     sample = trainer._prepare_sample(trainer._dummy_batch)
                    #     ignore_grad = True
                    # else:
                    #     ignore_grad = False

                    # no bos and eos
                    state, extra = trainer.model(sample['net_input']['src_tokens'],
                                                 features_only=True,
                                                 return_all_hiddens=True)
                    # print(sample['net_input']['src_tokens'])

                    head_num = args.decoder_attention_heads
                    state = state.reshape(state.size(0), state.size(1), head_num, state.size(-1) // head_num)
                    state = state.mean(2)

                    # last layer
                    # B T H
                    # state = states[select_layer - 1].transpose(0, 1)

                    for ib, l in enumerate(sample['net_input']['src_lengths']):
                        state_i = state[ib, :l].cpu().numpy()
                        all_hiddens.extend([si for si in state_i])

                    if len(all_hiddens) >= minibatch:
                        minibatch_hidden = np.stack(all_hiddens[:minibatch])
                        kmeans_model = kmeans_model.partial_fit(minibatch_hidden)
                        all_hiddens = all_hiddens[minibatch:]

                if len(all_hiddens) > 0:
                    minibatch_hidden = np.stack(all_hiddens)
                    kmeans_model = kmeans_model.partial_fit(minibatch_hidden)

                print('epc', epc)

    # print('n_seq, maxid', seq_num, maxid)
    print("(minibatch) kmeans consumes {} mins".format((time.time() - t1) // 60))

    # from joblib import dump, load
    # dump(kmeans_model, '{}/kmeans_k{}.joblib'.format(path, n_cluster))

    if not partial_fit:
        labels = kmeans_model.labels_
        print("labels + 1 to make label from 1, and 0 means padding")
        labels += 1
        print('labels shape', labels.shape)
        print('len(all_hiddens)', len(all_hiddens))
        print('sum(clusnum_perline)', sum(clusnum_perline))

    epoch_itr = trainer.get_train_iterator(
        epoch_itr.next_epoch_idx,
        # sharded data: get train iterator for next epoch
        load_dataset=task.has_sharded_data("train"),
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )

    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=False,
        shuffle=False,
    )

    print("assign labels")
    accu = 0
    # import _pickle as pck
    rand_index = []
    rand_sample_labels = []
    seq_num = 0

    for _, sample in enumerate(itr):
        sample, _ = trainer._prepare_sample(sample)

        if 'mt' in clustask:
            if clustask == 'mt-src':
                get_sample = sample['net_input']['src_tokens']
            else:
                get_sample = sample['target']

            for i, l in enumerate(get_sample):
                cluslabel_i = []
                clusnum_i = clusnum_perline[i]
                lable_ptr = 0

                for iw, w in enumerate(l):
                    if w.item() not in exwords:
                        cluslabel_i.append(labels[accu + lable_ptr])
                        lable_ptr += 1
                    else:
                        cluslabel_i.append(0)

                rand_sample_labels.append(cluslabel_i)

                accu += clusnum_i

                index = sample['id'][i].item()
                rand_index.append(index)
        else:
            seq_num += len(sample['net_input']['src_lengths'])

            for i, l in enumerate(sample['net_input']['src_lengths']):
                l = l.item()
                sample_labels = labels[accu: accu + l]
                rand_sample_labels.append(sample_labels)
                accu += l
                index = sample['id'][i].item()
                rand_index.append(index)

            print('seq_num', seq_num)
            print('len(seq_index), len(rand_sample_labels)', len(rand_index), len(rand_sample_labels))

    if onehead:
        write_name = '{}/{}.k{}.onehead.txt'.format(path, clustask, n_cluster)
    else:
        write_name = '{}/{}.k{}.txt'.format(path, clustask, n_cluster)

    with open(write_name, 'w') as fw:
        seq_index = np.argsort(rand_index)
        for i in seq_index:
            sample_label = list(map(str, rand_sample_labels[i]))
            sample_label = ' '.join(sample_label)
            print(sample_label, file=fw)

    # path = cfg.checkpoint.save_dir
    # pck.dump(word_center, open('{}/l{}k{}.pkl'.format(path, select_layer, n_cluster), 'wb'))


def cluster_sentence(cfg, trainer, task, epoch_itr):
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=False,
        set_dataset_epoch=False,
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    # itr = iterators.GroupedIterator(itr, update_freq)

    # trainer.begin_epoch(epoch_itr.epoch)

    should_stop = False

    # clustering parameters
    # select_layer = cfg.common.clus_layer
    n_cluster = cfg.common.n_cluster
    partial_fit = False
    onehead = False

    clustask = 'mt-src'
    # clustask = 'lm'

    # 0: embedding only
    feature_type = 1

    minibatch = 1024 * 20
    seed = cfg.common.seed
    # kmeans_model = MiniBatchKMeans(n_clusters=n_cluster, random_state=seed, batch_size=minibatch)
    kmeans_model = KMeans(n_clusters=n_cluster, random_state=seed)

    # for partial fit
    kmeans_epoch = 10

    # paths = utils.split_paths(args.data)
    # source_lang, target_lang = data_utils.infer_language_pair(paths[0])
    # src_dict = os.path.join(paths[0], "dict.txt".format(source_lang))
    # tgt_dict = os.path.join(paths[0], "dict.{}.txt".format(target_lang))

    def dict_freq(dict_path):
        # w_freq = {}
        w_freq = []
        for l in open(src_dict):
            w, freq = l.strip().split()

            if w in set(punctuation):
                continue
            # w_freq.append([w, freq])
            w_freq.append(task.dictionary.index(w))
            # w_freq[w] = int(freq)
        return w_freq

    # src_dict_freq = dict_freq(src_dict)
    path = cfg.checkpoint.save_dir

    t1 = time.time()
    # np.set_printoptions(threshold=sys.maxsize)
    trainer.model.eval()

    if not partial_fit:
        all_hiddens = []

        # itr = epoch_itr.next_epoch_itr(
        #     fix_batches_to_gpus=False,
        #     shuffle=False,
        # )

        with torch.no_grad():
            for i_s, sample in enumerate(itr):  # delayed update loop
                sample, _ = trainer._prepare_sample(sample)

                # no bos and eos
                if clustask == 'mt-src':
                    encoder_out = trainer.model.encoder(sample['net_input']['src_tokens'],
                                                        return_all_hiddens=True)

                    states = encoder_out['encoder_states']
                    # T B H
                    ctxstate = states[-1]

                    # sentence_rep
                    # B T
                    ctxstate = ctxstate.transpose(0, 1)
                    mask = encoder_out['encoder_padding_mask'][0]
                    ctxstate.masked_fill_(mask.unsqueeze(-1), 0)
                    # B H
                    state = ctxstate.sum(1) / sample['net_input']['src_lengths'].unsqueeze(-1)

                elif clustask == 'mt-tgt':
                    state, extra = trainer.model.lm_forward(**sample['net_input'],
                                                            features_only=True,
                                                            return_all_hiddens=True)
                elif clustask == 'lm':
                    state, extra = trainer.model(sample['net_input']['src_tokens'],
                                                 features_only=True,
                                                 return_all_hiddens=True)

                all_hiddens.append(state.cpu().numpy())

        # all_hiddens = np.stack(all_hiddens)
        all_hiddens = np.concatenate(all_hiddens, 0)
        print('all hidden shape', all_hiddens.shape)
        kmeans_model.fit(all_hiddens)

    # print('n_seq, maxid', seq_num, maxid)

    labels = kmeans_model.labels_
    print('labels shape', labels.shape)

    write_name = '{}/k{}.label'.format(path, n_cluster)
    with open(write_name, 'w', encoding='utf-8') as fw:
        for la in labels:
            print(la, file=fw)

    print("(minibatch) kmeans consumes {} mins".format((time.time() - t1) // 60))

    exit('over')

    # from joblib import dump, load
    # dump(kmeans_model, '{}/kmeans_k{}.joblib'.format(path, n_cluster))

    if not partial_fit:
        labels = kmeans_model.labels_
        print('labels shape', labels.shape)

    epoch_itr = trainer.get_train_iterator(
        epoch_itr.next_epoch_idx,
        # sharded data: get train iterator for next epoch
        load_dataset=task.has_sharded_data("train"),
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )

    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=False,
        shuffle=False,
    )

    print("assign labels")
    accu = 0
    # import _pickle as pck
    rand_index = []
    rand_sample_labels = []
    seq_num = 0

    # index = sample['id'][i].item()

    # for _, sample in enumerate(itr):
    #     sample, _ = trainer._prepare_sample(sample)
    #     if 'mt' in clustask:
    #         if clustask == 'mt-src':
    #             get_sample = sample['net_input']['src_tokens']
    #         else:
    #             get_sample = sample['target']

    write_name = '{}/{}.k{}.txt'.format(path, clustask, n_cluster)

    with open(write_name, 'w') as fw:
        seq_index = np.argsort(rand_index)
        for i in seq_index:
            sample_label = list(map(str, rand_sample_labels[i]))
            sample_label = ' '.join(sample_label)
            print(sample_label, file=fw)

    # path = cfg.checkpoint.save_dir
    # pck.dump(word_center, open('{}/l{}k{}.pkl'.format(path, select_layer, n_cluster), 'wb'))


def online_gau(cfg, trainer, task, epoch_itr):
    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=False,
        set_dataset_epoch=False,
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    # itr = iterators.GroupedIterator(itr, update_freq)

    # trainer.begin_epoch(epoch_itr.epoch)

    should_stop = False

    path = cfg.checkpoint.save_dir

    t1 = time.time()
    # np.set_printoptions(threshold=sys.maxsize)

    trainer.model.eval()
    trainer.model.start_mix()

    with torch.no_grad():
        for i_s, sample in enumerate(itr):  # delayed update loop
            sample, _ = trainer._prepare_sample(sample)
            encoder_out = trainer.model.encoder(sample['net_input']['src_tokens'],
                                                return_all_hiddens=True)

    torch.save(trainer.model.encoder.h_distru.state_dict(), f"{path}/hgau.pt")

    print("consumes {} mins".format((time.time() - t1) // 60))


def main(cfg: FairseqConfig) -> None:
    if isinstance(cfg, argparse.Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)

    utils.import_user_module(cfg.common)

    if distributed_utils.is_master(cfg.distributed_training) and "job_logging_cfg" in cfg:
        # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg))

    assert (
            cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"
    metrics.reset()

    if cfg.common.log_file is not None:
        handler = logging.FileHandler(filename=cfg.common.log_file)
        logger.addHandler(handler)

    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)

    if distributed_utils.is_master(cfg.distributed_training):
        checkpoint_utils.verify_checkpoint_directory(cfg.checkpoint.save_dir)

    # Print args
    logger.info(cfg)

    if cfg.checkpoint.write_checkpoints_asynchronously:
        try:
            import iopath  # noqa: F401
        except ImportError:
            logging.exception(
                "Asynchronous checkpoint writing is specified but iopath is "
                "not installed: `pip install iopath`"
            )
            return

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(cfg.task)

    assert cfg.criterion, "Please specify criterion to train a model"

    # Build model and criterion
    if cfg.distributed_training.ddp_backend == "fully_sharded":
        with fsdp_enable_wrap(cfg.distributed_training):
            model = fsdp_wrap(task.build_model(cfg.model))
    else:
        model = task.build_model(cfg.model)
    criterion = task.build_criterion(cfg.criterion)
    logger.info(model)
    logger.info("task: {}".format(task.__class__.__name__))
    logger.info("model: {}".format(model.__class__.__name__))
    logger.info("criterion: {}".format(criterion.__class__.__name__))
    logger.info(
        "num. shared model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if not getattr(p, "expert", False) and p.requires_grad)
        )
    )

    logger.info(
        "num. expert model params: {} (num. trained: {})".format(
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False)),
            sum(p.numel() for p in model.parameters() if getattr(p, "expert", False) and p.requires_grad),
        )
    )

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    # We load the valid dataset AFTER building the model
    data_utils.raise_if_valid_subsets_unintentionally_ignored(cfg)
    if cfg.dataset.combine_valid_subsets:
        task.load_dataset("valid", combine=True, epoch=1)
    else:
        for valid_sub_split in cfg.dataset.valid_subset.split(","):
            task.load_dataset(valid_sub_split, combine=False, epoch=1)

    # (optionally) Configure quantization
    if cfg.common.quantization_config_path is not None:
        quantizer = quantization_utils.Quantizer(
            config_path=cfg.common.quantization_config_path,
            max_epoch=cfg.optimization.max_epoch,
            max_update=cfg.optimization.max_update,
        )
    else:
        quantizer = None

    # Build trainer
    if cfg.common.model_parallel_size == 1:
        trainer = Trainer(cfg, task, model, criterion, quantizer)
    else:
        trainer = MegatronTrainer(cfg, task, model, criterion)
    logger.info(
        "training on {} devices (GPUs/TPUs)".format(
            cfg.distributed_training.distributed_world_size
        )
    )
    logger.info(
        "max tokens per device = {} and max sentences per device = {}".format(
            cfg.dataset.max_tokens,
            cfg.dataset.batch_size,
        )
    )

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
        cfg.checkpoint,
        trainer,
        # don't cache epoch iterators for sharded datasets
        disable_iterator_cache=task.has_sharded_data("train"),
    )
    if cfg.common.tpu:
        import torch_xla.core.xla_model as xm
        xm.rendezvous("load_checkpoint")  # wait for all workers

    max_epoch = cfg.optimization.max_epoch or math.inf
    lr = trainer.get_lr()

    train_meter = meters.StopwatchMeter()
    train_meter.start()

    best_loss = 100000
    while epoch_itr.next_epoch_idx <= max_epoch:
        if lr <= cfg.optimization.stop_min_lr:
            logger.info(
                f"stopping training because current learning rate ({lr}) is smaller "
                "than or equal to minimum learning rate "
                f"(--stop-min-lr={cfg.optimization.stop_min_lr})"
            )
            break

        # train for one epoch
        # print(type(epoch_itr))
        # print('epoch_itr.next_epoch_idx', epoch_itr.next_epoch_idx)
        # print(epoch_itr.epoch) start from 1
        # cluster_sentence(cfg, trainer, task, epoch_itr)
        online_gau(cfg, trainer, task, epoch_itr)

        # if should_stop:
        #     break

        # only use first validation loss to update the learning rate
        # lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        epoch_itr = trainer.get_train_iterator(
            epoch_itr.next_epoch_idx,
            # sharded data: get train iterator for next epoch
            load_dataset=task.has_sharded_data("train"),
            # don't cache epoch iterators for sharded datasets
            disable_iterator_cache=task.has_sharded_data("train"),
        )

    exit()

    train_meter.stop()
    logger.info("done training in {:.1f} seconds".format(train_meter.sum))

    # ioPath implementation to wait for all asynchronous file writes to complete.
    if cfg.checkpoint.write_checkpoints_asynchronously:
        logger.info(
            "ioPath PathManager waiting for all asynchronous checkpoint "
            "writes to finish."
        )
        PathManager.async_close()
        logger.info("ioPath PathManager finished waiting.")


def should_stop_early(cfg: DictConfig, valid_loss: float) -> bool:
    # skip check if no validation was done in the current epoch
    if valid_loss is None:
        return False
    if cfg.checkpoint.patience <= 0:
        return False

    def is_better(a, b):
        return a > b if cfg.checkpoint.maximize_best_checkpoint_metric else a < b

    prev_best = getattr(should_stop_early, "best", None)
    if prev_best is None or is_better(valid_loss, prev_best):
        should_stop_early.best = valid_loss
        should_stop_early.num_runs = 0
        return False
    else:
        should_stop_early.num_runs += 1
        if should_stop_early.num_runs >= cfg.checkpoint.patience:
            logger.info(
                "early stop since valid performance hasn't improved for last {} runs".format(
                    cfg.checkpoint.patience
                )
            )
            return True
        else:
            return False


@metrics.aggregate("train")
def train(
        cfg: DictConfig, trainer: Trainer, task: tasks.FairseqTask, epoch_itr
) -> Tuple[List[Optional[float]], bool]:
    """Train the model for one epoch and return validation losses."""
    # Initialize data iterator
    # debug
    # cluster_dict(cfg, trainer, task, epoch_itr)
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=cfg.distributed_training.fix_batches_to_gpus,
        shuffle=(epoch_itr.next_epoch_idx > cfg.dataset.curriculum),
    )
    update_freq = (
        cfg.optimization.update_freq[epoch_itr.epoch - 1]
        if epoch_itr.epoch <= len(cfg.optimization.update_freq)
        else cfg.optimization.update_freq[-1]
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    if cfg.common.tpu:
        itr = utils.tpu_data_loader(itr)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_file=cfg.common.log_file,
        log_interval=cfg.common.log_interval,
        epoch=epoch_itr.epoch,
        tensorboard_logdir=(
            cfg.common.tensorboard_logdir
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
        wandb_project=(
            cfg.common.wandb_project
            if distributed_utils.is_master(cfg.distributed_training)
            else None
        ),
        wandb_run_name=os.environ.get(
            "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
        ),
        azureml_logging=(
            cfg.common.azureml_logging
            if distributed_utils.is_master(cfg.distributed_training)
            else False
        ),
    )
    progress.update_config(_flatten_config(cfg))

    trainer.begin_epoch(epoch_itr.epoch)

    valid_subsets = cfg.dataset.valid_subset.split(",")
    should_stop = False
    num_updates = trainer.get_num_updates()
    logger.info("Start iterating over samples")
    for i, samples in enumerate(progress):
        with metrics.aggregate("train_inner"), torch.autograd.profiler.record_function(
                "train_step-%d" % i
        ):
            log_output = trainer.train_step(samples)
            # samples is a mini-batch

        if log_output is not None:  # not OOM, overflow, ...
            # log mid-epoch stats
            num_updates = trainer.get_num_updates()
            if num_updates % cfg.common.log_interval == 0:
                stats = get_training_stats(metrics.get_smoothed_values("train_inner"))
                progress.log(stats, tag="train_inner", step=num_updates)

                # reset mid-epoch stats after each log interval
                # the end-of-epoch stats will still be preserved
                metrics.reset_meters("train_inner")

        end_of_epoch = not itr.has_next()
        valid_losses, should_stop = validate_and_save(
            cfg, trainer, task, epoch_itr, valid_subsets, end_of_epoch
        )

        if should_stop:
            break

    # log end-of-epoch stats
    logger.info("end of epoch {} (average epoch stats below)".format(epoch_itr.epoch))
    stats = get_training_stats(metrics.get_smoothed_values("train"))
    progress.print(stats, tag="train", step=num_updates)

    # reset epoch-level meters
    metrics.reset_meters("train")
    return valid_losses, should_stop


def _flatten_config(cfg: DictConfig):
    config = OmegaConf.to_container(cfg)
    # remove any legacy Namespaces and replace with a single "args"
    namespace = None
    for k, v in list(config.items()):
        if isinstance(v, argparse.Namespace):
            namespace = v
            del config[k]
    if namespace is not None:
        config["args"] = vars(namespace)
    return config


def validate_and_save(
        cfg: DictConfig,
        trainer: Trainer,
        task: tasks.FairseqTask,
        epoch_itr,
        valid_subsets: List[str],
        end_of_epoch: bool,
) -> Tuple[List[Optional[float]], bool]:
    num_updates = trainer.get_num_updates()
    max_update = cfg.optimization.max_update or math.inf

    # Stopping conditions (and an additional one based on validation loss later
    # on)
    should_stop = False
    if num_updates >= max_update:
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"num_updates: {num_updates} >= max_update: {max_update}"
        )

    training_time_hours = trainer.cumulative_training_time() / (60 * 60)
    if (
            cfg.optimization.stop_time_hours > 0
            and training_time_hours > cfg.optimization.stop_time_hours
    ):
        should_stop = True
        logger.info(
            f"Stopping training due to "
            f"cumulative_training_time: {training_time_hours} > "
            f"stop_time_hours: {cfg.optimization.stop_time_hours} hour(s)"
        )

    do_save = (
            (end_of_epoch and epoch_itr.epoch % cfg.checkpoint.save_interval == 0)
            or should_stop
            or (
                    cfg.checkpoint.save_interval_updates > 0
                    and num_updates > 0
                    and num_updates % cfg.checkpoint.save_interval_updates == 0
                    and num_updates >= cfg.dataset.validate_after_updates
            )
    )
    do_validate = (
                          (not end_of_epoch and do_save)  # validate during mid-epoch saves
                          or (end_of_epoch and epoch_itr.epoch % cfg.dataset.validate_interval == 0)
                          or should_stop
                          or (
                                  cfg.dataset.validate_interval_updates > 0
                                  and num_updates > 0
                                  and num_updates % cfg.dataset.validate_interval_updates == 0
                          )
                  ) and not cfg.dataset.disable_validation and num_updates >= cfg.dataset.validate_after_updates

    # Validate
    valid_losses = [None]
    if do_validate:
        valid_losses = validate(cfg, trainer, task, epoch_itr, valid_subsets)

    should_stop |= should_stop_early(cfg, valid_losses[0])

    # Save checkpoint
    if do_save or should_stop:
        checkpoint_utils.save_checkpoint(
            cfg.checkpoint, trainer, epoch_itr, valid_losses[0]
        )

    return valid_losses, should_stop


def get_training_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    stats["wall"] = round(metrics.get_meter("default", "wall").elapsed_time, 0)
    return stats


def validate(
        cfg: DictConfig,
        trainer: Trainer,
        task: tasks.FairseqTask,
        epoch_itr,
        subsets: List[str],
) -> List[Optional[float]]:
    """Evaluate the model on the validation set(s) and return the losses."""

    if cfg.dataset.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(cfg.dataset.fixed_validation_seed)

    trainer.begin_valid_epoch(epoch_itr.epoch)
    valid_losses = []
    for subset in subsets:
        logger.info('begin validation on "{}" subset'.format(subset))

        # Initialize data iterator
        itr = trainer.get_valid_iterator(subset).next_epoch_itr(
            shuffle=False, set_dataset_epoch=False  # use a fixed valid set
        )
        if cfg.common.tpu:
            itr = utils.tpu_data_loader(itr)
        progress = progress_bar.progress_bar(
            itr,
            log_format=cfg.common.log_format,
            log_interval=cfg.common.log_interval,
            epoch=epoch_itr.epoch,
            prefix=f"valid on '{subset}' subset",
            tensorboard_logdir=(
                cfg.common.tensorboard_logdir
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
            wandb_project=(
                cfg.common.wandb_project
                if distributed_utils.is_master(cfg.distributed_training)
                else None
            ),
            wandb_run_name=os.environ.get(
                "WANDB_NAME", os.path.basename(cfg.checkpoint.save_dir)
            ),
        )

        # create a new root metrics aggregator so validation metrics
        # don't pollute other aggregators (e.g., train meters)
        with metrics.aggregate(new_root=True) as agg:
            for i, sample in enumerate(progress):
                if cfg.dataset.max_valid_steps is not None and i > cfg.dataset.max_valid_steps:
                    break
                trainer.valid_step(sample)

        # log validation stats
        stats = get_valid_stats(cfg, trainer, agg.get_smoothed_values())

        if hasattr(task, "post_validate"):
            task.post_validate(trainer.get_model(), stats, agg)

        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[cfg.checkpoint.best_checkpoint_metric])
    return valid_losses


def get_valid_stats(
        cfg: DictConfig, trainer: Trainer, stats: Dict[str, Any]
) -> Dict[str, Any]:
    stats["num_updates"] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, "best"):
        key = "best_{0}".format(cfg.checkpoint.best_checkpoint_metric)
        best_function = max if cfg.checkpoint.maximize_best_checkpoint_metric else min
        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            stats[cfg.checkpoint.best_checkpoint_metric],
        )
    return stats


def cli_main(
        modify_parser: Optional[Callable[[argparse.ArgumentParser], None]] = None
) -> None:
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser, modify_parser=modify_parser)

    cfg = convert_namespace_to_omegaconf(args)

    if cfg.common.use_plasma_view:
        server = PlasmaStore(path=cfg.common.plasma_path)
        logger.info(f"Started plasma server pid {server.server.pid} {cfg.common.plasma_path}")

    if args.profile:
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                distributed_utils.call_main(cfg, main)
    else:
        distributed_utils.call_main(cfg, main)

        # if cfg.common.use_plasma_view:
        #     server.server.kill()


if __name__ == "__main__":
    cli_main()
