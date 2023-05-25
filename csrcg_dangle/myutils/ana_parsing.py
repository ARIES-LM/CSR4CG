import numpy as np
from fairseq import checkpoint_utils, data, options, tasks, utils
from fairseq.logging import progress_bar
from fairseq.data import encoders
from fairseq.logging.meters import StopwatchMeter, TimeMeter

import torch
import sys, os, math, glob, re, random
from fairseq.data.encoders.gpt2_bpe_utils import get_encoder
from collections import Counter

# Parse command-line arguments for generation
parser = options.get_generation_parser(default_task='semantic_parsing')
args = options.parse_args_and_arch(parser)

# Setup task
task = tasks.setup_task(args)
task.load_dataset(args.gen_subset)

print(args)
print(args.max_sentences)
print(args.max_tokens)
# Set dictionaries
try:
    src_dict = getattr(task, 'source_dictionary', None)
except NotImplementedError:
    src_dict = None
tgt_dict = task.target_dictionary
src_dict = task.source_dictionary




import _pickle as pkl
import hypertools as hyp

# with open(args.results_path + '/hidden.pkl', 'wb') as fw:
#     pkl.dump(w_hs, fw)

# with open(args.results_path + 'hidden.pkl', 'rb') as fr:
#     w_hs = pkl.load(fr)
#
# words = ['in', 'on', 'beside']
# num = len(words)
#
# data = []
# labels = []
#
# for iw, w in enumerate(words):
#     data.append(np.stack(w_hs[w], 0))
#     labels.append([iw]*len(w_hs[w]))
#
# # data = np.stack(data, 0)
#
# hyp.plot([data[i] for i in range(num)], '.', ndims=2, save_path='{}/hymid.png'.format(args.results_path))
#
# exit()

# hue = [[dataid]*num for dataid in range(bsize)]
# hyp.plot(data, '.', hue=hue, save_path='{}/hyflathue.png'.format(path))




if args.results_path is not None:
    os.makedirs(args.results_path, exist_ok=True)
    output_path = os.path.join(args.results_path, 'generate-{}.txt'.format(args.gen_subset))
    output_file = open(output_path, 'w', buffering=1, encoding='utf-8')
else:
    output_file = sys.stdout

code_file_path = os.path.join(args.data, "{}.word-predicate.code".format(args.gen_subset))
if os.path.exists(code_file_path):
    with open(code_file_path) as f:
        codes = [line.strip() for line in f.readlines()]
else:
    codes = None


use_cuda = torch.cuda.is_available() and not args.cpu
# Load ensemble
print('loading model(s) from {}'.format(args.path))

# added
args.model_overrides = "{'encoder-embed-path':None, 'decoder-embed-path':None}"


models, _model_args = checkpoint_utils.load_model_ensemble(
    utils.split_paths(args.path),
    arg_overrides=eval(args.model_overrides),
    task=task,
    suffix=getattr(args, "checkpoint_suffix", ""),
)
# Optimize ensemble for generation
for model in models:
    model.prepare_for_inference_(args)
    if args.fp16:
        model.half()
    if use_cuda:
        model.cuda()

# Load dataset (possibly sharded)
itr = task.get_batch_iterator(
    dataset=task.dataset(args.gen_subset),
    max_tokens=args.max_tokens,
    max_sentences=args.max_sentences,
    max_positions=utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    ),
    ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
    required_batch_size_multiple=args.required_batch_size_multiple,
    num_shards=args.num_shards,
    shard_id=args.shard_id,
    num_workers=args.num_workers,
).next_epoch_itr(shuffle=False)
progress = progress_bar.progress_bar(
    itr,
    log_format=args.log_format,
    log_interval=args.log_interval,
    default_log_format=('tqdm' if not args.no_progress_bar else 'none'),
)


# Initialize generator
gen_timer = StopwatchMeter()
generator = task.build_generator(models, args)

# Handle tokenization and BPE
tokenizer = encoders.build_tokenizer(args)
bpe = encoders.build_bpe(args)
def decode_fn(x):
    if bpe is not None:
        x = bpe.decode(x)
    if tokenizer is not None:
        x = tokenizer.decode(x)
    return x

type_num, type_correct_num = Counter(), Counter()
correct, wrong, isum = 0, 0, 0
correct_examples, wrong_examples = [], []
gid = 0

w_hs = {}

for eid, sample in enumerate(progress):
    # print(eid)

    sample = utils.move_to_cuda(sample) if use_cuda else sample
    if 'net_input' not in sample:
        continue

    prefix_tokens = None
    if args.prefix_size > 0:
        prefix_tokens = sample['target'][:, :args.prefix_size]
    
    hypos = task.inference_step(generator, models, sample, prefix_tokens)

    gen_timer.start()

    # added
    with torch.no_grad():
        src_tokens = sample['net_input']['src_tokens']
        enc_h = models[0].encoder(src_tokens, src_lengths=sample['net_input']['src_lengths'])
        enc_h = enc_h.encoder_out.transpose(0, 1).cpu().numpy()

        # print(enc_h.shape)

        src_dictionary = models[0].encoder.dictionary
        # T B H
        for il, l in enumerate(src_tokens.tolist()):
            for iw, wid in enumerate(l):
                if wid in set(range(src_dictionary.nspecial)):
                    continue
                else:
                    w = src_dictionary[wid]
                    if w not in w_hs:
                        w_hs[w] = [enc_h[il, iw]]
                    else:
                        w_hs[w].append(enc_h[il, iw])

        import _pickle as pkl
        with open(args.results_path + '/hidden.pkl', 'wb') as fw:
            pkl.dump(w_hs, fw)




