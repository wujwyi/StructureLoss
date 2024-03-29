import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import logging

from transformers import AutoTokenizer, AutoModel, AutoConfig, T5ForConditionalGeneration, BartForConditionalGeneration,AutoModelForSeq2SeqLM
from geomloss import SamplesLoss
from utils import format_attention

logger = logging.getLogger(__name__)

MODEL_CHECKPOINTS = {'roberta': 'roberta-base',
                     'codebert': 'microsoft/codebert-base',
                     'graphcodebert': 'microsoft/graphcodebert-base',
                     't5': 't5-base',
                     'codet5': 'Salesforce/codet5-base',
                     'bart': 'facebook/bart-base',
                     'plbart': 'uclanlp/plbart-base'}


HUGGINGFACE_LOCALS = '../huggingface-models/'
MODEL_LOCALS = {
    'roberta': HUGGINGFACE_LOCALS + 'roberta-base',
    'codebert':  HUGGINGFACE_LOCALS + 'codebert-base',
    'graphcodebert':  HUGGINGFACE_LOCALS + 'graphcodebert-base',
    't5':  HUGGINGFACE_LOCALS + 't5-base',
    'codet5':  HUGGINGFACE_LOCALS + 'codet5-base',
    'codet5p-220m':  HUGGINGFACE_LOCALS + 'codet5p-220m',
    'codet5p-770m':  HUGGINGFACE_LOCALS + 'codet5p-770m',
    'bart':  HUGGINGFACE_LOCALS + 'bart-base',
    'plbart':  HUGGINGFACE_LOCALS + 'plbart-base',
    'unixcoder': HUGGINGFACE_LOCALS + 'unixcoder-base',
}


def calculate_attention_difference(model_attention, struc_attention, struc_loss_type=None, multi_head_loss=None):
    # model_attention_selected = model_attention[-1]  # choose the attention of last layer
    # print('model_attention', model_attention.shape)
    # model_attention:  Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).
    if multi_head_loss == 0:
        model_attention_selected = format_attention(
            model_attention, heads=0, layers=0)
        model_attention_selected = model_attention_selected.squeeze(
            dim=1)  # remove dimension for layers
        model_attention_selected = model_attention_selected.squeeze(
            dim=1)  # remove dimension for heads
        struc_attention = struc_attention.squeeze(dim=-1)
        # print('model_attention_selected shape', model_attention_selected.shape)
        # print('struc_attention shape', struc_attention.shape)
        matrix_size = struc_attention.shape[-1]
        if struc_loss_type == 'wasserstein':
            LOSS = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
            loss = LOSS(model_attention_selected, struc_attention)
            loss = loss.sum() / (matrix_size * matrix_size)
            # print('loss shape', loss.shape, ', loss', loss)
        return loss
    else:
        # get first layer, shape:(batch_size, num_heads, sequence_length, sequence_length)
        model_attention_selected = model_attention[0]

        struc_attention = struc_attention.squeeze(dim=-1)
        struc_attention = struc_attention.unsqueeze(1)

        num_head, seq_length = model_attention_selected.shape[1], struc_attention.shape[-1]

        struc_attention = struc_attention.repeat(1, num_head, 1, 1)

        model_attention_selected = model_attention_selected.permute(
            [1, 0, 2, 3]).contiguous()
        # model_attention_selected=model_attention_selected.transpose_(0, 1).contiguous()

        model_attention_selected = model_attention_selected.view(
            -1, model_attention_selected.shape[2], model_attention_selected.shape[3])  # shape:(batch_size*num_heads, sequence_length, sequence_length)

        struc_attention = struc_attention.view(
            -1, struc_attention.shape[2], struc_attention.shape[3])

        if struc_loss_type == 'wasserstein':
            LOSS = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
            loss = LOSS(model_attention_selected, struc_attention)
            loss = loss.sum() / (num_head * seq_length * seq_length)
            # print('loss shape', loss.shape, ', loss', loss)
        return loss


class StrucEncoder(nn.Module):
    def __init__(self, args=None):
        super(StrucEncoder, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, struc_feat):
        output = self.fc(struc_feat)
        return output


def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))


def bulid_or_load_gen_model(args):
    if 'sl' in args.model_name:
        model_name = args.model_name.strip('-sl')
    else:
        model_name = args.model_name

    # checkpoint = MODEL_LOCALS[model_name]
    checkpoint = MODEL_CHECKPOINTS[model_name]
    config = AutoConfig.from_pretrained(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    if 'summarize' in args.task:
        struc_encoder = StrucEncoder(args=args)
    else:
        if 'sl' in args.model_name:
            struc_encoder = StrucEncoder(args=args)

    if args.model_name in ['roberta', 'codebert', 'graphcodebert']:
        config.output_attentions = True
        encoder = AutoModel.from_pretrained(checkpoint, config=config)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model = Seq2Seq(
            encoder=encoder, decoder=decoder, config=config,
            beam_size=args.beam_size, max_length=args.max_target_length,
            sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id
        )
    elif args.model_name in ['t5', 'codet5']:
        config.output_attentions = True
        model = T5ForConditionalGeneration.from_pretrained(
            checkpoint, config=config)
    elif args.model_name in ['codet5p-220m','codet5p-770m']:
        config.output_attentions = True
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,
                                                  trust_remote_code=False,  # False for 220m and 770m models
                                                  torch_dtype=torch.float16,
                                                  low_cpu_mem_usage=True)
    elif args.model_name in ['codet5-220m', 'codet5-770m']:
        config.output_attentions = True
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,
                                                  trust_remote_code=False,  # False for 220m and 770m models
                                                  torch_dtype=torch.float16,
                                                  low_cpu_mem_usage=True)
    elif args.model_name in ['bart', 'plbart']:
        config.output_attentions = True
        model = BartForConditionalGeneration.from_pretrained(
            checkpoint, config=config)
    elif args.model_name in ['unixcoder']:
        config.is_decoder = True
        config.output_attentions = True
        encoder = AutoModel.from_pretrained(checkpoint, config=config)
        model = Seq2SeqforUnixcoder(
            encoder=encoder, decoder=encoder, config=config,
            beam_size=args.beam_size, max_length=args.max_target_length,
            sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],
            eos_id=tokenizer.sep_token_id
        )
    elif args.model_name in ['unixcoder-sl']:
        config.is_decoder = True
        config.output_attentions = True
        encoder = AutoModel.from_pretrained(checkpoint, config=config)
        model = Seq2SeqforUnixcoderWithSL(
            encoder=encoder, decoder=encoder, config=config,struc_encoder=struc_encoder,
            beam_size=args.beam_size, max_length=args.max_target_length,
            sos_id=tokenizer.convert_tokens_to_ids(["<mask0>"])[0],
            eos_id=tokenizer.sep_token_id
        )
    elif args.model_name in ['codet5-sl']:
        config.output_attentions = True
        t5_model = T5ForConditionalGeneration.from_pretrained(
            checkpoint, config=config)
        model = Codet5WithSL(t5_model=t5_model, struc_encoder=struc_encoder)
    elif args.model_name in ['codet5-220m-sl', 'codet5-770m-sl']:
        config.output_attentions = True
        t5_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint,
                                                  trust_remote_code=False,  # False for 220m and 770m models
                                                  torch_dtype=torch.float16,
                                                  low_cpu_mem_usage=True)
        model = Codet5WithSL(t5_model=t5_model, struc_encoder=struc_encoder)
    elif args.model_name in ['roberta-sl', 'codebert-sl', 'graphcodebert-sl']:
        config.output_attentions = True
        encoder = AutoModel.from_pretrained(checkpoint, config=config)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model = Seq2SeqWithSL(
            encoder=encoder, decoder=decoder, struc_encoder=struc_encoder, config=config,
            beam_size=args.beam_size, max_length=args.max_target_length,
            sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id,
        )

    logger.info("Finish loading model [%s] parameters from %s", get_model_size(
        model), args.model_name)

    return config, model, tokenizer


# https://github.com/microsoft/CodeBERT/blob/master/CodeBERT/code2nl/model.py
class Seq2Seq(nn.Module):
    """
        Build Seqence-to-Sequence.

        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model.
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
    """

    def __init__(self, encoder, decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)

    def forward(self, source_ids=None, source_mask=None, target_ids=None, target_mask=None, args=None):
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_attention = outputs[-1]
        encoder_output = outputs[0].permute([1, 0, 2]).contiguous()
        if target_ids is not None:
            attn_mask = -1e4 * \
                (1 - self.bias[:target_ids.shape[1], :target_ids.shape[1]])
            tgt_embeddings = self.encoder.embeddings(
                target_ids).permute([1, 0, 2]).contiguous()
            out = self.decoder(tgt_embeddings, encoder_output, tgt_mask=attn_mask,
                               memory_key_padding_mask=~source_mask)
            # memory_key_padding_mask=(1 - source_mask).bool())
            hidden_states = torch.tanh(self.dense(
                out)).permute([1, 0, 2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])
            struc_loss = torch.tensor(0.0, device=loss.device)
            return loss, struc_loss, loss * active_loss.sum(), active_loss.sum(), encoder_attention
        else:
            # Predict
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(source_ids.shape[0]):
                context = encoder_output[:, i:i + 1]
                context_mask = source_mask[i:i + 1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_length):
                    if beam.done():
                        break
                    attn_mask = -1e4 * \
                        (1 - self.bias[:input_ids.shape[1],
                         :input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embeddings(
                        input_ids).permute([1, 0, 2]).contiguous()
                    out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask,
                                       memory_key_padding_mask=~context_mask)
                    # memory_key_padding_mask=(1 - context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute(
                        [1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(
                        0, beam.getCurrentOrigin()))
                    input_ids = torch.cat(
                        (input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
                        pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            preds = torch.cat(preds, 0)
            return preds, encoder_attention


class Codet5WithSL(nn.Module):
    def __init__(self, t5_model, struc_encoder):
        super(Codet5WithSL, self).__init__()
        self.t5_model = t5_model
        self.struc_encoder = struc_encoder

    def forward(self, input_ids, attention_mask, labels=None, decoder_attention_mask=None, sl_feats=None, args=None):
        if sl_feats is not None:
            # print('before sl_feats shape', sl_feats.shape)
            sl_feats = sl_feats.view(
                input_ids.shape[0], args.max_source_length, args.max_source_length, -1)
            # print('after sl_feats shape', sl_feats.shape)
        if labels is not None:
            output = self.t5_model(input_ids=input_ids, attention_mask=attention_mask,
                                   labels=labels, decoder_attention_mask=decoder_attention_mask)
            encoder_attention = output.encoder_attentions
            loss = output.loss
            if sl_feats is not None:
                struc_attention = self.struc_encoder(sl_feats)
                struc_loss = calculate_attention_difference(
                    encoder_attention, struc_attention, struc_loss_type=args.struc_loss_type, multi_head_loss=args.multi_head_loss)
            else:
                struc_loss = torch.tensor(0.0, device=loss.device)
            return loss, struc_loss

    def generate(self, source_ids, attention_mask, use_cache, num_beams, early_stopping, max_length):
        return self.t5_model.generate(source_ids, attention_mask=attention_mask, use_cache=use_cache, num_beams=num_beams, early_stopping=early_stopping, max_length=max_length)


class Seq2SeqWithSL(nn.Module):
    """
        Build Seqence-to-Sequence.

        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model.
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
    """

    def __init__(self, encoder, decoder, config, struc_encoder=None, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2SeqWithSL, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.struc_encoder = struc_encoder
        self.register_buffer("bias", torch.tril(torch.ones(2048, 2048)))
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)
        self.lsm = nn.LogSoftmax(dim=-1)
        self.tie_weights()

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.encoder.embeddings.word_embeddings)

    def forward(self, source_ids=None, source_mask=None, target_ids=None, target_mask=None, args=None, sl_feats=None):
        if sl_feats is not None:
            # print('before sl_feats shape', sl_feats.shape)
            sl_feats = sl_feats.view(
                source_ids.shape[0], args.max_source_length, args.max_source_length, -1)
            # print('after sl_feats shape', sl_feats.shape)
        outputs = self.encoder(source_ids, attention_mask=source_mask)
        encoder_attention = outputs[-1]
        encoder_output = outputs[0].permute([1, 0, 2]).contiguous()
        if target_ids is not None:
            attn_mask = -1e4 * \
                (1 - self.bias[:target_ids.shape[1], :target_ids.shape[1]])
            tgt_embeddings = self.encoder.embeddings(
                target_ids).permute([1, 0, 2]).contiguous()
            out = self.decoder(tgt_embeddings, encoder_output, tgt_mask=attn_mask,
                               memory_key_padding_mask=~source_mask)
            # memory_key_padding_mask=(1 - source_mask).bool())
            hidden_states = torch.tanh(self.dense(
                out)).permute([1, 0, 2]).contiguous()
            lm_logits = self.lm_head(hidden_states)
            # Shift so that tokens < n predict n
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                            shift_labels.view(-1)[active_loss])
            if sl_feats is not None:
                struc_attention = self.struc_encoder(sl_feats)
                struc_loss = calculate_attention_difference(
                    encoder_attention, struc_attention, struc_loss_type=args.struc_loss_type, multi_head_loss=args.multi_head_loss)
            else:
                struc_loss = torch.tensor(0.0, device=loss.device)
            return loss, struc_loss, loss * active_loss.sum(), active_loss.sum(), encoder_attention
        else:
            # Predict
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(source_ids.shape[0]):
                context = encoder_output[:, i:i + 1]
                context_mask = source_mask[i:i + 1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_length):
                    if beam.done():
                        break
                    attn_mask = -1e4 * \
                        (1 - self.bias[:input_ids.shape[1],
                         :input_ids.shape[1]])
                    tgt_embeddings = self.encoder.embeddings(
                        input_ids).permute([1, 0, 2]).contiguous()
                    out = self.decoder(tgt_embeddings, context, tgt_mask=attn_mask,
                                       memory_key_padding_mask=~context_mask)
                    # memory_key_padding_mask=(1 - context_mask).bool())
                    out = torch.tanh(self.dense(out))
                    hidden_states = out.permute(
                        [1, 0, 2]).contiguous()[:, -1, :]
                    out = self.lsm(self.lm_head(hidden_states)).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(
                        0, beam.getCurrentOrigin()))
                    input_ids = torch.cat(
                        (input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
                        pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            preds = torch.cat(preds, 0)
            return preds, encoder_attention


# https://github.com/microsoft/CodeBERT/blob/master/UniXcoder/downstream-tasks/code-summarization/model.py
class Seq2SeqforUnixcoder(nn.Module):
    """
        Build Seqence-to-Sequence.

        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """

    def __init__(self, encoder, decoder, config, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2SeqforUnixcoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.register_buffer(
            "bias", torch.tril(torch.ones(
                (1024, 1024), dtype=torch.uint8)).view(1, 1024, 1024)
        )
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        self.lsm = nn.LogSoftmax(dim=-1)

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def forward(self, source_ids, target_ids=None):
        if target_ids is None:
            return self.generate(source_ids)

        mask = source_ids.ne(1)[:, None, :]*source_ids.ne(1)[:, :, None]
        encoder_output = self.encoder(
            source_ids, attention_mask=mask, use_cache=True)
        encoder_attention = encoder_output[-1]
        ids = torch.cat((source_ids, target_ids), -1)
        mask = self.bias[:,
                         source_ids.size(-1):ids.size(-1), :ids.size(-1)].bool()
        mask = mask & ids[:, None, :].ne(1)

        out = self.decoder(target_ids, attention_mask=mask,
                           past_key_values=encoder_output.past_key_values).last_hidden_state
        lm_logits = self.lm_head(out)
        # Shift so that tokens < n predict n
        active_loss = target_ids[..., 1:].ne(1).view(-1)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                        shift_labels.view(-1)[active_loss])

        outputs = loss, loss*active_loss.sum(), active_loss.sum(), encoder_attention
        return outputs

    def generate(self, source_ids):
        mask = source_ids.ne(1)[:, None, :]*source_ids.ne(1)[:, :, None]
        encoder_output = self.encoder(
            source_ids, attention_mask=mask, use_cache=True)
        encoder_attention = encoder_output[-1]
        preds = []
        zero = torch.cuda.LongTensor(1).fill_(0)
        source_len = list(source_ids.ne(1).sum(-1).cpu().numpy())
        for i in range(source_ids.shape[0]):
            context = [[x[i:i+1, :, :source_len[i]].repeat(self.beam_size, 1, 1, 1) for x in y]
                       for y in encoder_output.past_key_values]
            beam = Beam(self.beam_size, self.sos_id, self.eos_id)
            input_ids = beam.getCurrentState()
            context_ids = source_ids[i:i+1,
                                     :source_len[i]].repeat(self.beam_size, 1)
            for _ in range(self.max_length):
                if beam.done():
                    break

                ids = torch.cat((context_ids, input_ids), -1)
                mask = self.bias[:,
                                 context_ids.size(-1):ids.size(-1), :ids.size(-1)].bool()
                mask = mask & ids[:, None, :].ne(1)
                out = self.decoder(input_ids, attention_mask=mask,
                                   past_key_values=context).last_hidden_state
                hidden_states = out[:, -1, :]
                out = self.lsm(self.lm_head(hidden_states)).data
                beam.advance(out)
                input_ids.data.copy_(input_ids.data.index_select(
                    0, beam.getCurrentOrigin()))
                input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
            hyp = beam.getHyp(beam.getFinal())
            pred = beam.buildTargetTokens(hyp)[:self.beam_size]
            pred = [torch.cat([x.view(-1) for x in p]+[zero] *
                              (self.max_length-len(p))).view(1, -1) for p in pred]
            preds.append(torch.cat(pred, 0).unsqueeze(0))

        preds = torch.cat(preds, 0)

        return preds, encoder_attention

class Seq2SeqforUnixcoderWithSL(nn.Module):
    """
        Build Seqence-to-Sequence.

        Parameters:

        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model. 
        * `beam_size`- beam size for beam search. 
        * `max_length`- max length of target for beam search. 
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search. 
    """

    def __init__(self, encoder, decoder, config, struc_encoder=None, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Seq2SeqforUnixcoderWithSL, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.struc_encoder = struc_encoder
        self.register_buffer(
            "bias", torch.tril(torch.ones(
                (1024, 1024), dtype=torch.uint8)).view(1, 1024, 1024)
        )
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.encoder.embeddings.word_embeddings.weight
        self.lsm = nn.LogSoftmax(dim=-1)

        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id

    def forward(self, source_ids, target_ids=None, args=None, sl_feats=None):
        if sl_feats is not None:
            # print('before sl_feats shape', sl_feats.shape)
            sl_feats = sl_feats.view(
                source_ids.shape[0], args.max_source_length, args.max_source_length, -1)
            # print('after sl_feats shape', sl_feats.shape)
        if target_ids is None:
            return self.generate(source_ids)

        mask = source_ids.ne(1)[:, None, :]*source_ids.ne(1)[:, :, None]
        encoder_output = self.encoder(
            source_ids, attention_mask=mask, use_cache=True)
        encoder_attention = encoder_output[-1]
        ids = torch.cat((source_ids, target_ids), -1)
        mask = self.bias[:,
                         source_ids.size(-1):ids.size(-1), :ids.size(-1)].bool()
        mask = mask & ids[:, None, :].ne(1)

        out = self.decoder(target_ids, attention_mask=mask,
                           past_key_values=encoder_output.past_key_values).last_hidden_state
        lm_logits = self.lm_head(out)
        # Shift so that tokens < n predict n
        active_loss = target_ids[..., 1:].ne(1).view(-1)
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                        shift_labels.view(-1)[active_loss])
        if sl_feats is not None:
                struc_attention = self.struc_encoder(sl_feats)
                struc_loss = calculate_attention_difference(
                    encoder_attention, struc_attention, struc_loss_type=args.struc_loss_type, multi_head_loss=args.multi_head_loss)
        else:
            struc_loss = torch.tensor(0.0, device=loss.device)
        outputs = loss, struc_loss, loss*active_loss.sum(), active_loss.sum(), encoder_attention
        return outputs

    def generate(self, source_ids):
        mask = source_ids.ne(1)[:, None, :]*source_ids.ne(1)[:, :, None]
        encoder_output = self.encoder(
            source_ids, attention_mask=mask, use_cache=True)
        encoder_attention = encoder_output[-1]
        preds = []
        zero = torch.cuda.LongTensor(1).fill_(0)
        source_len = list(source_ids.ne(1).sum(-1).cpu().numpy())
        for i in range(source_ids.shape[0]):
            context = [[x[i:i+1, :, :source_len[i]].repeat(self.beam_size, 1, 1, 1) for x in y]
                       for y in encoder_output.past_key_values]
            beam = Beam(self.beam_size, self.sos_id, self.eos_id)
            input_ids = beam.getCurrentState()
            context_ids = source_ids[i:i+1,
                                     :source_len[i]].repeat(self.beam_size, 1)
            for _ in range(self.max_length):
                if beam.done():
                    break

                ids = torch.cat((context_ids, input_ids), -1)
                mask = self.bias[:,
                                 context_ids.size(-1):ids.size(-1), :ids.size(-1)].bool()
                mask = mask & ids[:, None, :].ne(1)
                out = self.decoder(input_ids, attention_mask=mask,
                                   past_key_values=context).last_hidden_state
                hidden_states = out[:, -1, :]
                out = self.lsm(self.lm_head(hidden_states)).data
                beam.advance(out)
                input_ids.data.copy_(input_ids.data.index_select(
                    0, beam.getCurrentOrigin()))
                input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
            hyp = beam.getHyp(beam.getFinal())
            pred = beam.buildTargetTokens(hyp)[:self.beam_size]
            pred = [torch.cat([x.view(-1) for x in p]+[zero] *
                              (self.max_length-len(p))).view(1, -1) for p in pred]
            preds.append(torch.cat(pred, 0).unsqueeze(0))

        preds = torch.cat(preds, 0)

        return preds, encoder_attention

class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                           .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.

        Parameters:

        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step

        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
