TOP_N=$1
LABEL_FIELD=$2
HIDDEN_DIM=$3
USE_NODE_VECTOR=$4
EMB=$5 # "all" or "combined"

EMBEDDING_DIM=384
GLOVE_DIM=300
INPUT_DIM=300

DATA_ROOT="/shared/1/projects/concite/"

TRAINING_DATA="$DATA_ROOT"data/acl_data/train_"$TOP_N"_"$LABEL_FIELD"_acl_data.jsonl
DEV_DATA="$DATA_ROOT"data/acl_data/dev_"$TOP_N"_"$LABEL_FIELD"_acl_data.jsonl
TEST_DATA="$DATA_ROOT"data/acl_data/test_"$TOP_N"_"$LABEL_FIELD"_acl_data.jsonl

echo $TRAINING_DATA

export LABEL_FIELD=$LABEL_FIELD
export USE_NODE_VECTOR=$USE_NODE_VECTOR
export TOP_N=$TOP_N
export HIDDEN_DIM=$HIDDEN_DIM
export GLOVE_WEIGHTS=/home/khenner/glove/glove.840B.300d.zip
export TRAINING_DATA=$TRAINING_DATA
export DEV_DATA=$DEV_DATA
export TEST_DATA=$TEST_DATA

SERIALIZATION_DIR=/shared/2/projects/concite/serialization/"$LABEL_FIELD"_serialization/"$SEED"/model_glove

if [ "$USE_NODE_VECTOR" == "true" ]; then
  SERIALIZATION_DIR="$SERIALIZATION_DIR"_n2v_"$EMB"
  (( INPUT_DIM = INPUT_DIM + EMBEDDING_DIM ))
  export PRETRAINED_FILE="$DATA_ROOT"data/embeddings/"$EMB".emb
else
  export PRETRAINED_FILE=None
fi

export INPUT_DIM=$INPUT_DIM
export EMBEDDING_DIM=$EMBEDDING_DIM

echo Serialization DIR: "$SERIALIZATION_DIR"
echo Vector FIle: "$PRETRAINED_FILE"
echo INPUT_DIM: "$INPUT_DIM"
echo EMBEDDing_DIM: "$EMBEDDING_DIM"
allennlp train allennlp_configs/acl_glove_classifier.json -s $SERIALIZATION_DIR -f --include-package concite
