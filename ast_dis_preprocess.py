import argparse
import networkx as nx
import sys
import os
import torch

from transformers import AutoTokenizer
from tqdm import tqdm
from utils import read_summarize_examples, format_special_chars, get_filenames, traverse, index_to_code_token
from tree_sitter import Language, Parser

sys.setrecursionlimit(5000)


# MODEL_CHECKPOINTS = {'roberta': 'roberta-base',
#                      'codebert': 'microsoft/codebert-base',
#                      'graphcodebert': 'microsoft/graphcodebert-base',
#                      't5': 't5-base',
#                      'codet5': 'Salesforce/codet5-base',
#                      'bart': 'facebook/bart-base',
#                      'plbart': 'uclanlp/plbart-base'}

HUGGINGFACE_LOCALS = '../huggingface-models/'
MODEL_LOCALS = {
    'roberta': HUGGINGFACE_LOCALS + 'roberta-base',
    'codebert':  HUGGINGFACE_LOCALS + 'codebert-base',
    'graphcodebert':  HUGGINGFACE_LOCALS + 'graphcodebert-base',
    't5':  HUGGINGFACE_LOCALS + 't5-base',
    'codet5':  HUGGINGFACE_LOCALS + 'codet5-base',
    'bart':  HUGGINGFACE_LOCALS + 'bart-base',
    'plbart':  HUGGINGFACE_LOCALS + 'plbart-base',
    'unixcoder': HUGGINGFACE_LOCALS + 'unixcoder-base',
}


def get_subtokens(source_code, tokenizer, max_length):
    source_ids = tokenizer.encode(
        source_code, max_length=max_length, padding='max_length', truncation=True)
    subtokens = []
    for source_id in source_ids:
        subtoken = tokenizer.convert_ids_to_tokens(source_id)
        subtokens.append(subtoken)
    subtokens = format_special_chars(subtokens)
    return subtokens


def get_traverse_graph(source_code, lang):
    LANGUAGE = Language('evaluator/CodeBLEU/parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    tree = parser.parse(bytes(source_code, 'utf-8'))
    cursor = tree.walk()
    G = nx.Graph()
    traverse(cursor, G, came_up=False, node_tag=0, node_sum=0, parent_dict={})
    return G


def get_shortest_path_length_in_tree(G):
    return dict(nx.shortest_path_length(G))


def get_T_token_number_type(G, source_code):
    T = nx.dfs_tree(G, 0)
    leaves = [x for x in T.nodes() if T.out_degree(x) ==
              0 and T.in_degree(x) == 1]
    token_number_dict = {}
    tokens_type_dict = {}
    for leaf in leaves:
        feature = G.nodes[leaf]['features']
        if feature.type != 'comment':
            start = feature.start_point
            end = feature.end_point
            token = index_to_code_token([start, end], source_code)
            token_number_dict[leaf] = token
            tokens_type_dict[leaf] = feature.type
    return T, token_number_dict, tokens_type_dict


def get_token_map_subtoken(subtokens, tokens, tokens_number, tokenizer):
    tokens_pos = 0
    token_map_list = []
    token_map_dict = {}
    token_cum = ''
    for j in range(len(subtokens)):
        if subtokens[j] in ['<s>', '</s>', '<pad>'] or subtokens[j] in tokenizer.additional_special_tokens:
            pass
        else:
            token_cum += subtokens[j]
            token_map_list.append(j)
            # print('token_cum:', token_cum, ', tokens_pos:', tokens_pos, 'token', tokens[tokens_pos])
            # delete all null str
            while tokens_pos < len(tokens) and tokens[tokens_pos] == '':
                tokens_pos += 1
            if tokens_pos >= len(tokens):
                return token_map_dict  # handle the exception
            if token_cum == tokens[tokens_pos] or j == len(subtokens) - 1:
                token_map_dict[tokens_number[tokens_pos]] = token_map_list
                token_map_list = []
                token_cum = ''
                tokens_pos += 1
    return token_map_dict


def get_subtoken_map_token(token_map_dict):
    # whether to add the postion of subtoken
    subtoken_map_dict = {}
    for token in token_map_dict:
        for subtoken in token_map_dict[token]:
            if subtoken not in subtoken_map_dict:
                subtoken_map_dict[subtoken] = token
    return subtoken_map_dict


def get_sast(T, leaves, tokens_dict, tokens_type_dict):
    # add subtoken edges and Data flow edges to T
    T = nx.Graph(T)
    subtoken_edges = []
    dataflow_edges = []
    identifier_dict = {}
    i = 0
    for leaf in leaves:
        token = tokens_dict[leaf]
        token_type = tokens_type_dict[leaf]
        if token_type == 'identifier':
            if token not in identifier_dict:
                identifier_dict[token] = leaf
            else:
                dataflow_edges.append((identifier_dict[token], leaf))
                identifier_dict[token] = leaf
        if i > 0:
            subtoken_edges.append((old_leaf, leaf))
        old_leaf = leaf
        i += 1
    T.add_edges_from(subtoken_edges)
    T.add_edges_from(dataflow_edges)
    return T  # new_T


def generate_ast_dis(filename, tokenizer, args):
    data_all = read_summarize_examples(filename=filename, data_num=-1)
    u_ast_tag = ''
    if args.upgraded_ast:
        u_ast_tag = '_uast'
    if 'train' in filename:
        target_dir = filename.replace(
            'train.jsonl', '{}/train/'.format(args.model_name + '-sl' + u_ast_tag))
    elif 'valid' in filename:
        target_dir = filename.replace(
            'valid.jsonl', '{}/valid/'.format(args.model_name + '-sl' + u_ast_tag))
    else:
        target_dir = filename.replace(
            'test.jsonl', '{}/test/'.format(args.model_name + '-sl' + u_ast_tag))
    print('target_dir', target_dir)
    max_length = args.max_length
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for data in tqdm(data_all):
        source_code, idx = data.source, data.idx
        subtokens = get_subtokens(
            source_code=source_code, tokenizer=tokenizer, max_length=args.max_length)
        G = get_traverse_graph(source_code=source_code, lang=args.lang)
        # token_number_dict:    key:Nodes number in the ast tree,    value:token in source code
        T, token_number_dict, tokens_type_dict = get_T_token_number_type(
            G=G, source_code=source_code)
        u_ast = get_sast(T, token_number_dict.keys(),
                         token_number_dict, tokens_type_dict)
        tokens, tokens_number = list(
            token_number_dict.values()), list(token_number_dict.keys())
        # token_map_dict:    key:Nodes number in the ast tree,   value:index in subtokens
        token_map_dict = get_token_map_subtoken(
            subtokens=subtokens, tokens=tokens, tokens_number=tokens_number, tokenizer=tokenizer)
        # subtoken_map_dict:    key:index in subtokens,   value:Nodes number in the ast tree
        subtoken_map_dict = get_subtoken_map_token(token_map_dict)
        if args.upgraded_ast:
            shortest_path_length = get_shortest_path_length_in_tree(u_ast)
        else:
            shortest_path_length = get_shortest_path_length_in_tree(G)
        target_name = '{}/{}.pt'.format(target_dir, idx)
        dis_mat = torch.zeros(max_length, max_length)
        for i in range(max_length):
            for j in range(max_length):
                if i in subtoken_map_dict and j in subtoken_map_dict:
                    dis_mat[i][j] = shortest_path_length[subtoken_map_dict[i]
                                                         ][subtoken_map_dict[j]]
        torch.save(dis_mat, target_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="roberta",
                        type=str, choices=['roberta', 'codebert', 'graphcodebert', 'bart', 'plbart', 't5', 'codet5', 'unixcoder'])
    parser.add_argument("--task", type=str, required=True,
                        choices=['summarize-idx'])
    parser.add_argument("--sub_task", type=str, default='')
    parser.add_argument("--lang", type=str, default='')
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--use_shortest_path", action='store_true')
    parser.add_argument("--upgraded_ast", action='store_true')
    args = parser.parse_args()

    if args.task == 'summarize-idx':
        args.lang = args.sub_task

    checkpoint = MODEL_LOCALS[args.model_name]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    filenames = get_filenames(data_root='/mnt/e/data',
                              task=args.task, sub_task=args.sub_task)
    print('filenames', filenames)
    for fn in filenames:
        generate_ast_dis(filename=fn, tokenizer=tokenizer, args=args)


if __name__ == "__main__":
    main()


