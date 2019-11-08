TOP_N=$1
LABEL_FIELD=$2
TEXT_FIELD=$3
HIDDEN_DIM=$4
USE_TEXT=$5
USE_NODE_VECTOR=$6
EMB_TYPE=$7 # "all" or "combined"
EMB_L=$8
EMB_P=$9
EMB_Q=${10}
INTENT_WT=${11}

EMBEDDING_DIM=384
COMBINED_SIZE=1536

DATA_ROOT="/shared-1/projects/concite/"

PAPER_LOOKUP_PATH="$DATA_ROOT"data/acl_data/acl_data.jsonl

TRAINING_DATA="$DATA_ROOT"data/acl_data/edge_data/train_abstract_acl_edge_data.jsonl
DEV_DATA="$DATA_ROOT"data/acl_data/edge_data/dev_abstract_acl_edge_data.jsonl
TEST_DATA="$DATA_ROOT"data/acl_data/edge_data/test_abstract_acl_edge_data.jsonl

echo $TRAINING_DATA

SERIALIZATION_DIR_NAME=serialization_"$HIDDEN_DIM"

echo $LABEL_FIELD
echo $TEXT_FIELD

export LABEL_FIELD=$LABEL_FIELD
export TEXT_FIELD=$TEXT_FIELD
export USE_TEXT=$USE_TEXT
export USE_NODE_VECTOR=$USE_NODE_VECTOR
export HIDDEN_DIM=$HIDDEN_DIM
export BERT_VOCAB=/home/khenner/scibert_scivocab_uncased/vocab.txt
export BERT_WEIGHTS=/home/khenner/scibert_scivocab_uncased/weights.tar.gz
export TRAINING_DATA=$TRAINING_DATA
export DEV_DATA=$DEV_DATA
export TEST_DATA=$TEST_DATA
export PAPER_LOOKUP_PATH=$PAPER_LOOKUP_PATH
export TOP_N=$TOP_N

EMB_SUFFIX="$EMB_TYPE"_"$EMB_L"_"$EMBEDDING_DIM"_"$EMB_P"_"$EMB_Q"
if [ "$EMB_TYPE" == "combined" ]; then
  EMB_SUFFIX="$EMB_SUFFIX"_"$INTENT_WT"
fi

SERIALIZATION_DIR="$DATA_ROOT""$SERIALIZATION_DIR_NAME"/"$LABEL_FIELD"

if [ "$USE_TEXT" == "true" ]; then
    SERIALIZATION_DIR="$SERIALIZATION_DIR"_"$TEXT_FIELD"
fi

if [ "$USE_NODE_VECTOR" == "true" ]; then
    SERIALIZATION_DIR="$SERIALIZATION_DIR"_n2v_"$EMB_SUFFIX"
fi

if [ "$USE_NODE_VECTOR" == "true" ]; then
  export PRETRAINED_FILE="$DATA_ROOT"data/embeddings/"$EMB_SUFFIX".emb
else
  export PRETRAINED_FILE=None
fi

export EMBEDDING_DIM=$EMBEDDING_DIM
export INPUT_DIM=$COMBINED_SIZE

echo Serialization DIR: "$SERIALIZATION_DIR"
echo Vector FIle: "$PRETRAINED_FILE"
allennlp train allennlp_configs/acl_edge_classifier.json -s $SERIALIZATION_DIR -f --include-package concite &&
rm "$SERIALIZATION_DIR"/training_state_epoch_*
rm "$SERIALIZATION_DIR"/model_state_epoch_*
