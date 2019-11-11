TOP_N=$1
LABEL_FIELD=$2
HIDDEN_DIM=$3
USE_ABSTRACT=$4
USE_NODE_VECTOR=$5
EMB_TYPE=$6 # "all" or "combined"
EMB_L=$7
EMB_P=$8
EMB_Q=$9
INTENT_WT=${10}

EMBEDDING_DIM=384
BERT_DIM=768
INPUT_DIM=0

DATA_ROOT="/shared-1/projects/concite/"

TRAINING_DATA="$DATA_ROOT"data/acl_data/train_"$TOP_N"_"$LABEL_FIELD"_acl_data.jsonl
DEV_DATA="$DATA_ROOT"data/acl_data/dev_"$TOP_N"_"$LABEL_FIELD"_acl_data.jsonl
TEST_DATA="$DATA_ROOT"data/acl_data/test_"$TOP_N"_"$LABEL_FIELD"_acl_data.jsonl

echo $TRAINING_DATA

SERIALIZATION_DIR_NAME=serialization_"$HIDDEN_DIM"

export LABEL_FIELD=$LABEL_FIELD
export USE_ABSTRACT=$USE_ABSTRACT
export USE_NODE_VECTOR=$USE_NODE_VECTOR
export TOP_N=$TOP_N
export HIDDEN_DIM=$HIDDEN_DIM
export EMBEDDING_DIM=$EMBEDDING_DIM
export BERT_VOCAB=/home/khenner/scibert_scivocab_uncased/vocab.txt
export BERT_WEIGHTS=/home/khenner/scibert_scivocab_uncased/weights.tar.gz
export TRAINING_DATA=$TRAINING_DATA
export DEV_DATA=$DEV_DATA
export TEST_DATA=$TEST_DATA

EMB_SUFFIX="$EMB_TYPE"_"$EMB_L"_"$EMBEDDING_DIM"_"$EMB_P"_"$EMB_Q"
if [ "$EMB_TYPE" == "combined" ]; then
  EMB_SUFFIX="$EMB_SUFFIX"_"$INTENT_WT"
fi

SERIALIZATION_DIR="$DATA_ROOT""$SERIALIZATION_DIR_NAME"/"$TOP_N"_"$LABEL_FIELD"

if [ "$USE_ABSTRACT" == "true" ]; then
    SERIALIZATION_DIR="$SERIALIZATION_DIR"_abstract
    (( INPUT_DIM = INPUT_DIM + BERT_DIM ))
fi

if [ "$USE_NODE_VECTOR" == "true" ]; then
  SERIALIZATION_DIR="$SERIALIZATION_DIR"_n2v_"$EMB_SUFFIX"
  (( INPUT_DIM = INPUT_DIM + EMBEDDING_DIM ))
  export PRETRAINED_FILE="$DATA_ROOT"data/embeddings/"$EMB_SUFFIX".emb
else
  export PRETRAINED_FILE=None
fi

export INPUT_DIM=$INPUT_DIM

echo Serialization DIR: "$SERIALIZATION_DIR"
echo Vector FIle: "$PRETRAINED_FILE"
echo $INPUT_DIM
allennlp train allennlp_configs/acl_classifier.json -s $SERIALIZATION_DIR -f --include-package concite &&
rm "$SERIALIZATION_DIR"/training_state_epoch_*
rm "$SERIALIZATION_DIR"/model_state_epoch_*
