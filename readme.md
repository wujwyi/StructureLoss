# SAT

## Setup

This implemetation is based on Python 3.8. To run the code, you need the following dependencies:

- tree_sitter==0.2.1
- transformers==4.27.1
- torch=1.14.0+cu118
- networkx
- tensorboard
- geomloss

Or you can simply run:

```python
pip install -r requirements.txt
```

## Repository structure

```python
|-- main.py # The main function to run the code
|-- utils.py # The preprocess for all datasets
|-- models.py # The key models in our experiments
|-- write_idx_to_dataset.py # Pre-process the raw data and change its format
|-- ast_dis_preprocess.py # Parse the code into ast and generate the distance matrix
|-- configs.py # Hyperparameters of the experiment


```

## Download Datasets

We download the [CodeXGLUE](https://github.com/microsoft/CodeXGLUE) dataset for code summarization from [link](https://github.com/microsoft/CodeXGLUE/blob/main/Code-Text/code-to-text/dataset.zip). And the dataset for code translation from [link](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Code/code-to-code-trans/data).


## Run pipeline


1.Place the dataset in a folder named "data" at the same level as main.py

2.Generate the distance matrixes. Note that if do code summarization task, maxlength is 256
```shell
python3 ast_dis_preprocess.py --model_name codebert --task translate-idx --sub_task cs-java --max_length 320
```

3.Make predictions for downstream tasks.

For example, for java-cs# translation tasks with codebert. The meaning of each token can be found in run.sh. In this example, the first 0 represents gpu0 and 1e-1 represents the weight of the structure loss is 1e-1, 1.0 represents the data set sampling rate
```shell
bash run.sh codebert-sl translate-idx java-cs 0 1e-1 1.0
```


## Attribution

Parts of this code are based on the following repositories:

- [CodeT5](https://github.com/salesforce/CodeT5)

