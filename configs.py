import random
import torch
import logging
import numpy as np
import multiprocessing

logger = logging.getLogger(__name__)


def add_args(parser):
    parser.add_argument("--task", type=str, required=True,
                        choices=['summarize', 'summarize-idx', 'refine', 'translate', 'translate-idx', 'concode', 'clone', 'defect'])
    parser.add_argument("--sub_task", type=str, default='')
    parser.add_argument("--lang", type=str, default='')
    parser.add_argument("--add_lang_ids", action='store_true')
    # plbart unfinished
    parser.add_argument("--model_name", default="roberta",
                        type=str, choices=['roberta', 'codebert', 'graphcodebert', 'bart', 'plbart', 't5', 'codet5', 'codet5p-220m', 'codet5p-770m', 'unixcoder',
                                           'roberta-sl', 'codebert-sl', 'graphcodebert-sl', 'unixcoder-sl', 'codet5-sl'])
    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed for initialization")  # previous one 42
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument("--cache_path", type=str, default='cache_data')
    parser.add_argument("--res_dir", type=str, default='results',
                        help='directory to save fine-tuning results')
    parser.add_argument("--res_fn", type=str, default='')
    parser.add_argument("--model_dir", type=str, default='saved_models',
                        help='directory to save fine-tuned models')
    parser.add_argument("--summary_dir", type=str, default='tensorboard',
                        help='directory to save tensorboard summary')
    parser.add_argument("--data_num", type=int, default=-1,
                        help='number of data instances to use, -1 for full data')
    parser.add_argument("--gpu", type=int, default=0,
                        help='index of the gpu to use in a cluster')
    parser.add_argument("--data_dir", default='data', type=str)
    parser.add_argument("--output_dir", default='outputs', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--add_task_prefix", action='store_true',
                        help="Whether to add task prefix for t5 and codet5")
    parser.add_argument("--save_last_checkpoints", type=int, default=0)
    parser.add_argument("--always_save_model", action='store_true')
    parser.add_argument("--always_remove_model", type=int, default=1)
    parser.add_argument("--do_eval_bleu", action='store_true',
                        help="Whether to evaluate bleu on dev set.")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_train_epochs", default=100, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--struc_loss_type", type=str, default='wasserstein')
    parser.add_argument("--alpha", type=float, default=1e-4,
                        help='Parameter to balance structure loss')
    parser.add_argument("--use_sumppl_in_struc_eval", action='store_true')
    parser.add_argument("--multi_head_loss", type=int, default=0)
    parser.add_argument("--upgraded_ast", type=int, default=0)
    parser.add_argument("--debug_mode", type=int, default=0)
    parser.add_argument("--early_patience", type=int, default=1)
    parser.add_argument("--sample_rate", type=float, default=2.0)
    args = parser.parse_args()
    return args


def set_dist(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Setup for distributed data parallel
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    cpu_count = multiprocessing.cpu_count()
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), cpu_count)
    args.device = device
    args.cpu_count = cpu_count


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def set_hyperparas(args):
    if args.task in ['summarize', 'summarize-idx']:
        args.adam_epsilon = 1e-8
        args.beam_size = 10
        args.gradient_accumulation_steps = 1
        args.lr = 5e-5
        args.max_source_length = 256
        args.max_target_length = 128
        args.num_train_epochs = 100
        # args.num_train_epochs = 15 # will not early stop on some datasets if the number of training epochs is too small
        args.patience = 5
        args.weight_decay = 0.0
        args.warmup_steps = 1000
        args.lang = args.sub_task

        if args.model_name in ['roberta', 'codebert', 'graphcodebert']:
            # args.batch_size = 128 # A100
            args.batch_size = 48  # V100
        elif args.model_name in ['t5', 'codet5']:
            # args.batch_size = 64 # A100
            args.batch_size = 32
        elif args.model_name in ['bart', 'plbart']:
            # args.batch_size = 128 # A100
            args.batch_size = 48  # V100
        elif args.model_name in ['unixcoder']:
            # args.batch_size = 128 # A100
            args.batch_size = 40  # V100
            args.early_patience = 0
            args.patience = 2
            # args.gradient_accumulation_steps = 2
        elif args.model_name in ['roberta-sl', 'codebert-sl', 'graphcodebert-sl']:
            # args.batch_size = 128 # A100
            args.batch_size = 44  # V100
            args.is_sl = True
            # args.data_num = 1000
        elif args.model_name in ['unixcoder-sl']:
            # args.batch_size = 128 # A100
            args.batch_size = 38  # V100
            args.is_sl = True
            args.early_patience = 0
            args.patience = 2
            if args.sub_task=='go':
                args.batch_size = 36
            # args.gradient_accumulation_steps = 2
        elif args.model_name in ['codet5-sl']:
            # args.batch_size = 128 # A100
            args.batch_size = 48  # V100
            args.patience = 2
            args.is_sl = True
            args.early_patience = 0
        elif args.model_name in ['codet5p-220m','codet5p-770m']:
            # args.batch_size = 128 # A100
            args.batch_size = 48  # V100
            args.patience = 2
            args.is_sl = True
            args.early_patience = 0

    elif args.task in ['translate','translate-idx']:
        args.adam_epsilon = 1e-8
        args.beam_size = 10
        args.gradient_accumulation_steps = 1
        args.lr = 2e-5
        args.max_source_length = 320
        args.max_target_length = 256
        args.num_train_epochs = 50
        args.patience = 20
        args.weight_decay = 0.0
        args.warmup_steps = 1000

        if args.sub_task == 'java-cs':
            args.lang = 'c_sharp'
        elif args.sub_task == 'cs-java':
            args.lang = 'java'

        if args.model_name in ['roberta', 'codebert', 'graphcodebert','roberta-sl', 'codebert-sl', 'graphcodebert-sl']:
            # args.batch_size = 128  # A100
            args.batch_size = 32  # V100
        elif args.model_name in ['t5', 'codet5', 'codet5-sl']:
            # args.batch_size = 64  # A100
            args.batch_size = 16
        elif args.model_name in ['bart', 'plbart']:
            # args.batch_size = 128  # A100
            args.batch_size = 48  # V100
        elif args.model_name in ['unixcoder','unixcoder-sl']:
            # args.batch_size = 128  # A100
            args.batch_size = 48  # V100

    if args.debug_mode:
        args.batch_size = 16
