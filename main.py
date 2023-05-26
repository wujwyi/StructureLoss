import logging
import torch
import argparse
import time
import multiprocessing
import os
import numpy as np
import math
import shutil

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from configs import add_args, set_dist, set_seed, set_hyperparas
from models import bulid_or_load_gen_model
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data, SummarizeDataset, read_examples
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size,
                                 num_workers=4, pin_memory=True)
    # Start evaluating model
    logger.info("  " + "***** Running ppl evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.batch_size)

    model.eval()
    eval_loss, eval_struc_loss, batch_num = 0, 0, 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval ppl"):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            if args.model_name in ['roberta', 'codebert', 'graphcodebert', 'roberta-sl',
                                   'codebert-sl', 'graphcodebert-sl']:
                loss, struc_loss, _, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                                  target_ids=target_ids, target_mask=target_mask)
            elif args.model_name in ['unixcoder']:
                loss, _, _, _ = model(
                    source_ids=source_ids, target_ids=target_ids)
                struc_loss = torch.tensor(0.0, device=loss.device)
            elif args.model_name in ['unixcoder-sl']:
                loss, struc_loss, _, _, _ = model(
                    source_ids=source_ids, target_ids=target_ids)
                struc_loss = torch.tensor(0.0, device=loss.device)
            else:
                loss, struc_loss = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                struc_loss = torch.tensor(0.0, device=loss.device)

        eval_loss += loss.item()
        eval_struc_loss += struc_loss.item()
        batch_num += 1
    eval_loss = eval_loss / batch_num
    eval_struc_loss = eval_struc_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    eval_struc_ppl = round(np.exp(eval_struc_loss), 5)
    return eval_ppl, eval_struc_ppl


def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    logger.info(
        "  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(
            eval_data, sampler=eval_sampler, batch_size=args.batch_size)

    model.eval()
    pred_ids = []
    bleu, codebleu = 0.0, 0.0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            if args.model_name in ['roberta', 'codebert', 'graphcodebert', 'roberta-sl',
                                   'codebert-sl', 'graphcodebert-sl']:
                preds, _ = model(source_ids=source_ids,
                                 source_mask=source_mask)

                top_preds = [pred[0].cpu().numpy() for pred in preds]
            elif args.model_name in ['unixcoder', 'unixcoder-sl']:
                preds, _ = model(source_ids=source_ids)
                top_preds = [pred[0].cpu().numpy() for pred in preds]
            else:
                if 'summarize' in args.task:
                    early_stopping = True
                else:
                    early_stopping = False
                preds = model.generate(source_ids,
                                       attention_mask=source_mask,
                                       use_cache=True,
                                       num_beams=args.beam_size,
                                       early_stopping=early_stopping,
                                       max_length=args.max_target_length)
                top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)
    # pdb.set_trace()
    pred_nls = [tokenizer.decode(
        id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]

    output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
    gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
    src_fn = os.path.join(args.res_dir, "test_{}.src".format(criteria))

    if args.task in ['defect']:
        target_dict = {0: 'false', 1: 'true'}
        golds = [target_dict[ex.target] for ex in eval_examples]
        eval_acc = np.mean([int(p == g) for p, g in zip(pred_nls, golds)])
        result = {'em': eval_acc, 'bleu': 0, 'codebleu': 0}

        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                f.write(pred_nl.strip() + '\n')
                f1.write(target_dict[gold.target] + '\n')
                f2.write(gold.source.strip() + '\n')
            logger.info("Save the predictions into %s", output_fn)
    else:
        dev_accs, predictions = [], []
        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                dev_accs.append(pred_nl.strip() == gold.target.strip())
                if 'summarize' in args.task:
                    predictions.append(str(gold.idx) + '\t' + pred_nl)
                    f.write(str(gold.idx) + '\t' +
                            pred_nl.strip().encode('utf8').decode() + '\n')
                    f1.write(str(gold.idx) + '\t' +
                             gold.target.strip().encode('utf8').decode() + '\n')
                    f2.write(str(gold.idx) + '\t' +
                             gold.source.strip().encode('utf8').decode() + '\n')
                else:
                    f.write(pred_nl.strip().encode('utf8').decode() + '\n')
                    f1.write(gold.target.strip().encode(
                        'utf8').decode() + '\n')
                    f2.write(gold.source.strip().encode(
                        'utf8').decode() + '\n')

        if 'summarize' in args.task:
            (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
            bleu = round(smooth_bleu.bleuFromMaps(
                goldMap, predictionMap)[0], 2)
        else:
            bleu = round(_bleu(gold_fn, output_fn), 2)
            if split_tag == 'test' and 'translate' in args.task:
                codebleu = calc_code_bleu.get_codebleu(
                    gold_fn, output_fn, args.lang)
        # except:
        #     bleu = 0.0
        #     codebleu = 0.0

        em = np.mean(dev_accs) * 100
        result = {'em': em, 'bleu': bleu}
        if not args.task == 'summarize' and split_tag == 'test':
            result['codebleu'] = codebleu * 100

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def main():
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    t0 = time.time()

    set_dist(args)
    set_seed(args)
    set_hyperparas(args)

    logger.info(args)

    if 'summarize' in args.task or 'translate' in args.task:
        config, model, tokenizer = bulid_or_load_gen_model(args)

    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(args.cpu_count)
    args.train_filename, args.dev_filename, args.test_filename = get_filenames(
        args.data_dir, args.task, args.sub_task)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir,
                                        '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        if 'sl' not in args.model_name:
            # Prepare training data loader
            train_examples, train_data = load_and_cache_gen_data(
                args, args.train_filename, pool, tokenizer, 'train')
            tmp = len(train_data)
            train_sampler = RandomSampler(
                train_data) if args.local_rank == -1 else DistributedSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size,
                                          num_workers=10, pin_memory=True)
        else:
            train_examples = read_examples(
                args.train_filename, args.data_num, args.task)
            train_data = SummarizeDataset(
                examples=train_examples,
                tokenizer=tokenizer,
                args=args,
                stage='train',
                only_src=False
            )
            tmp = len(train_data)
            train_sampler = RandomSampler(
                train_data) if args.local_rank == -1 else DistributedSampler(train_data)
            train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size,
                                          num_workers=4, pin_memory=True)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.lr, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * \
            len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.batch_size * args.gradient_accumulation_steps)
        logger.info("  Batch num = %d", math.ceil(
            train_example_num / args.batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        dev_dataset = {}
        global_step, best_bleu_em, best_ppl = 0, -1, 1e6
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6
        first_not_inc_cnt_3_flag=0

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(
                train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss, tr_struc_loss = 0, 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(args.device) for t in batch)
                if 'sl' in args.model_name:
                    source_ids, target_ids, sl_feats = batch
                else:
                    source_ids, target_ids = batch
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                target_mask = target_ids.ne(tokenizer.pad_token_id)

                if args.model_name in ['roberta', 'codebert', 'graphcodebert']:
                    loss, struc_loss, _, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                                      target_ids=target_ids, target_mask=target_mask)
                elif args.model_name in ['unixcoder']:
                    loss, _, _, _ = model(
                        source_ids=source_ids, target_ids=target_ids)
                    struc_loss = torch.tensor(0.0, device=loss.device)
                elif args.model_name in ['codet5', 'plbart']:
                    outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                    labels=target_ids, decoder_attention_mask=target_mask)
                    loss = outputs.loss
                    struc_loss = torch.tensor(0.0, device=loss.device)
                elif args.model_name in ['codet5-sl']:
                    loss, struc_loss = model(input_ids=source_ids, attention_mask=source_mask,
                                             labels=target_ids, decoder_attention_mask=target_mask, sl_feats=sl_feats, args=args)
                    loss = loss + struc_loss * args.alpha
                elif args.model_name in ['roberta-sl', 'codebert-sl', 'graphcodebert-sl', 'codet5-sl']:
                    loss, struc_loss, _, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                                      target_ids=target_ids, target_mask=target_mask,
                                                      sl_feats=sl_feats, args=args)
                    loss = loss + struc_loss * args.alpha
                elif args.model_name in ['unixcoder-sl']:
                    loss, struc_loss, _, _, _ = model(
                        source_ids=source_ids, target_ids=target_ids,sl_feats=sl_feats, args=args)
                    loss = loss + struc_loss * args.alpha

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                    struc_loss = struc_loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    struc_loss = struc_loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                tr_struc_loss += struc_loss.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(
                        tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    train_struc_loss = round(
                        tr_struc_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    bar.set_description("[{}] Train loss {}, Train struc loss {}".format(
                        cur_epoch, round(train_loss, 3), round(train_struc_loss, 3)))

            if args.do_eval:
                # Eval model with dev dataset
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples, eval_data = load_and_cache_gen_data(
                        args, args.dev_filename, pool, tokenizer, 'dev')
                    dev_dataset['dev_loss'] = eval_examples, eval_data

                eval_ppl, eval_struc_ppl = eval_ppl_epoch(
                    args, eval_data, eval_examples, model, tokenizer)
                eval_ppl_sum = eval_ppl + eval_struc_ppl * args.alpha
                result = {'epoch': cur_epoch,  'global_step': global_step, 'train_loss:': train_loss,
                          'train_struc_loss': train_struc_loss, 'eval_ppl': eval_ppl,
                          'eval_struc_ppl': eval_struc_ppl, 'eval_ppl_sum': eval_ppl_sum}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)
                if args.data_num == -1:
                    tb_writer.add_scalar('train_loss', train_loss, cur_epoch)
                    tb_writer.add_scalar(
                        'train_struc_loss', train_struc_loss, cur_epoch)
                    tb_writer.add_scalar('dev_ppl', eval_ppl_sum, cur_epoch)
                    tb_writer.add_scalar(
                        'dev_struc_ppl', eval_struc_ppl, cur_epoch)

                # save last checkpoint
                if args.save_last_checkpoints:
                    last_output_dir = os.path.join(
                        args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save = model.module if hasattr(
                        model, 'module') else model
                    output_model_file = os.path.join(
                        last_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the last model into %s",
                                output_model_file)

                if args.use_sumppl_in_struc_eval:
                    eval_ppl = eval_ppl_sum

                if eval_ppl < best_ppl:
                    not_loss_dec_cnt = 0
                    logger.info("  Best ppl:%s", eval_ppl)
                    logger.info("  " + "*" * 20)
                    fa.write("[%d] Best ppl changed into %.4f\n" %
                             (cur_epoch, eval_ppl))
                    best_ppl = eval_ppl

                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(
                        args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if args.always_save_model:
                        model_to_save = model.module if hasattr(
                            model, 'module') else model
                        output_model_file = os.path.join(
                            output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(),
                                   output_model_file)
                        logger.info(
                            "Save the best ppl model into %s", output_model_file)
                else:
                    not_loss_dec_cnt += 1

                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                if args.do_eval_bleu:
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev',
                                                                       only_src=True, is_sample=True)

                    result = eval_bleu_epoch(
                        args, eval_data, eval_examples, model, tokenizer, 'dev', 'e%d' % cur_epoch)
                    dev_bleu, dev_em = result['bleu'], result['em']
                    if args.task in ['summarize', 'summarize-idx']:
                        dev_bleu_em = dev_bleu
                    elif args.task in ['defect']:
                        dev_bleu_em = dev_em
                    else:
                        dev_bleu_em = dev_bleu + dev_em
                    if args.data_num == -1:
                        tb_writer.add_scalar(
                            'dev_bleu_em', dev_bleu_em, cur_epoch)
                        # tb_writer.add_scalar('dev_em', dev_em, cur_epoch)
                    if dev_bleu_em > best_bleu_em:
                        not_bleu_em_inc_cnt = 0
                        logger.info("  [%d] Best bleu+em: %.2f (bleu: %.2f, em: %.2f)",
                                    cur_epoch, dev_bleu_em, dev_bleu, dev_em)
                        logger.info("  " + "*" * 20)
                        best_bleu_em = dev_bleu_em
                        fa.write("[%d] Best bleu+em changed into %.2f (bleu: %.2f, em: %.2f)\n" % (
                            cur_epoch, best_bleu_em, dev_bleu, dev_em))
                        # Save best checkpoint for best bleu
                        output_dir = os.path.join(
                            args.output_dir, 'checkpoint-best-bleu')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or args.always_save_model:
                            model_to_save = model.module if hasattr(
                                model, 'module') else model
                            output_model_file = os.path.join(
                                output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(),
                                       output_model_file)
                            logger.info(
                                "Save the best bleu model into %s", output_model_file)
                    else:
                        not_bleu_em_inc_cnt += 1
                        logger.info(
                            "Bleu does not increase for %d epochs", not_bleu_em_inc_cnt)
                        if args.early_patience == 1: 
                            if not_bleu_em_inc_cnt > 2 and not_loss_dec_cnt > 2  and first_not_inc_cnt_3_flag == 0:
                                first_not_inc_cnt_3_flag = 1
                                output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu-patience3')
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                src_path=os.path.join(args.output_dir, 'checkpoint-best-bleu/pytorch_model.bin')
                                dst_path=output_dir+'/pytorch_model.bin'
                                shutil.copy(src_path, dst_path)
                                logger.info("Save the patience3 best bleu model into %s", dst_path)

                                output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl-patience3')
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                src_path=os.path.join(args.output_dir, 'checkpoint-best-ppl/pytorch_model.bin')
                                dst_path=output_dir+'/pytorch_model.bin'
                                shutil.copy(src_path, dst_path)
                                logger.info("Save the patience3 best ppl model into %s", dst_path)

                                output_dir = os.path.join(args.output_dir, 'checkpoint-last-patience3')
                                if not os.path.exists(output_dir):
                                    os.makedirs(output_dir)
                                src_path=os.path.join(args.output_dir, 'checkpoint-last/pytorch_model.bin')
                                dst_path=output_dir+'/pytorch_model.bin'
                                shutil.copy(src_path, dst_path)
                                logger.info("Save the patience3 best ppl model into %s", dst_path)
                        if not_bleu_em_inc_cnt > args.patience and not_loss_dec_cnt > 2:
                            stop_early_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                                cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                            logger.info(stop_early_str)
                            fa.write(stop_early_str)
                            break
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.batch_size)
        
        if args.early_patience == 0:
            checkpoint_list=['best-bleu', 'best-ppl', 'last']
        else:
            checkpoint_list=['best-bleu', 'best-ppl', 'last','best-bleu-patience3', 'best-ppl-patience3', 'last-patience3']
        for criteria in checkpoint_list:  # 'best-bleu', 'best-ppl', 'last'  # 'best-bleu', 'best-ppl', 'last'
            file = os.path.join(
                args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            logger.info("Reload model from {}".format(file))
            model.load_state_dict(torch.load(file))
            eval_examples, eval_data = load_and_cache_gen_data(args, args.test_filename, pool, tokenizer, 'test',
                                                               only_src=True, is_sample=False)
            result = eval_bleu_epoch(
                args, eval_data, eval_examples, model, tokenizer, 'test', criteria)
            test_bleu, test_em = result['bleu'], result['em']
            test_codebleu = result['codebleu'] if 'codebleu' in result else 0
            result_str = "[%s] bleu-4: %.2f, em: %.4f, codebleu: %.4f\n" % (
                criteria, test_bleu, test_em, test_codebleu)
            logger.info(result_str)
            fa.write(result_str)
            if args.res_fn:
                with open(args.res_fn, 'a+') as f:
                    f.write('[Time: {}] {}\n'.format(
                        get_elapse_time(t0), file))
                    f.write(result_str)

    if args.always_remove_model:
        if args.early_patience == 0:
            checkpoint_list=['best-bleu', 'best-ppl', 'last']
        else:
            checkpoint_list=['best-bleu', 'best-ppl', 'last','best-bleu-patience3', 'best-ppl-patience3', 'last-patience3']
        for criteria in checkpoint_list:  # 'best-bleu', 'best-ppl', 'last'  # 'best-bleu', 'best-ppl', 'last'
            file = os.path.join(
                args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            dir_path = os.path.join(
                args.output_dir, 'checkpoint-{}'.format(criteria))
            os.remove(file)
            os.rmdir(dir_path)
    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()


if __name__ == "__main__":
    main()
