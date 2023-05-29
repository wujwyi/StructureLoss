WORKDIR="./"
export PYTHONPATH=$WORKDIR

MODEL_NAME=${1}
TASK=${2}
SUB_TASK=${3}
CUDA_NO=${4}
ALPHA=${5}
STORE=1
REMOVE=1
SEED=1234
MULTIHEADLOSS=${6}
UAST=${7}
SAMPLE=${8}
TESTTAG=${9}



DATA_NUM=-1
MODEL_DIR=save_models
SUMMARY_DIR=tensorboard
FULL_MODEL_TAG=${MODEL_NAME}

if [[ ${SUB_TASK} == none ]]; then
  OUTPUT_DIR=${MODEL_DIR}/${TASK}/${FULL_MODEL_TAG}
  RES_DIR=results/${TASK}/${FULL_MODEL_TAG}
  RES_FN=results/${TASK}/${FULL_MODEL_TAG}.txt
else
  OUTPUT_DIR=${MODEL_DIR}/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}/${ALPHA}/${TESTTAG}
  RES_DIR=results/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}/${ALPHA}/${TESTTAG}
  RES_FN=results/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}/${ALPHA}/${TESTTAG}/${FULL_MODEL_TAG}.txt
fi

CACHE_DIR=${OUTPUT_DIR}/cache_data
LOG=${OUTPUT_DIR}/train.log
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}
mkdir -p ${RES_DIR}

RUN_FN=${WORKDIR}/main.py

CUDA_VISIBLE_DEVICES=${CUDA_NO} \
TOKENIZERS_PARALLELISM=false \
  python3.8 -u ${RUN_FN} ${MULTI_TASK_AUG} \
  --do_test --do_train --do_eval --do_eval_bleu --save_last_checkpoints ${STORE} --always_save_model --always_remove_model ${REMOVE} \
  --task ${TASK} --sub_task ${SUB_TASK} --model_name ${MODEL_NAME} --data_num ${DATA_NUM}  \
  --output_dir ${OUTPUT_DIR}  --summary_dir ${SUMMARY_DIR} \
  --data_dir /mnt/e/data  --cache_path ${CACHE_DIR} --res_dir ${RES_DIR} --res_fn ${RES_FN} --use_sumppl_in_struc_eval\
  --alpha ${ALPHA} --seed ${SEED} --multi_head_loss ${MULTIHEADLOSS} --upgraded_ast ${UAST} --sample_rate ${SAMPLE}\
  2>&1 | tee ${LOG}
