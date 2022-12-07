import json
import os

class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 code_tokens,
                 docstring_tokens,
                 code
                 ):
        self.idx = idx
        self.code_tokens = code_tokens
        self.docstring_tokens = docstring_tokens
        self.code = code


def read_summarize_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            examples.append(
                Example(
                    idx=idx,
                    code=js['code'],
                    code_tokens=js['code_tokens'],
                    docstring_tokens=js['docstring_tokens']
                )
            )
            if idx + 1 == data_num:
                break
    return examples

def main():
    tasks = ['summarize']
    origin_root = './data'
    for task in tasks:
        if task == 'summarize':
            langs = ['java', 'go', 'javascript', 'php', 'python', 'ruby']
            # langs = ['ruby']
            for lang in langs:
                data_dir = '{}/{}/{}'.format(origin_root, task, lang)
                target_dir = data_dir.replace('summarize', 'summarize-idx')
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                train_fn = '{}/train.jsonl'.format(data_dir)
                dev_fn = '{}/valid.jsonl'.format(data_dir)
                test_fn = '{}/test.jsonl'.format(data_dir)
                fns = [train_fn, dev_fn, test_fn]
                for fn in fns:
                    data = read_summarize_examples(filename=fn, data_num=-1)
                    target_fn = fn.replace('summarize', 'summarize-idx')
                    json_file = open(target_fn, 'w')
                    for d in data:
                        json_str = json.dumps(d.__dict__)
                        json_file.write(json_str)
                        json_file.write('\n')
                    json_file.close()
                    print('write {} finished'.format(target_fn))
                    
if __name__ == "__main__":
    main()
