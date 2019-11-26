TOP_N=$1
LABEL_FIELD=$2
HIDDEN_DIM=$3
USE_ABSTRACT=$4
USE_NODE_VECTOR=$5
EMB=$6

EMBEDDING_DIM=384
BERT_DIM=768
INPUT_DIM=0

DATA_ROOT="/shared-1/projects/concite/"

TRAINING_DATA="$DATA_ROOT"data/acl_data/train_"$TOP_N"_"$LABEL_FIELD"_acl_data.jsonl
DEV_DATA="$DATA_ROOT"data/acl_data/dev_"$TOP_N"_"$LABEL_FIELD"_acl_data.jsonl
TEST_DATA="$DATA_ROOT"data/acl_data/test_"$TOP_N"_"$LABEL_FIELD"_acl_data.jsonl

echo $TRAINING_DATA

export LABEL_FIELD=$LABEL_FIELD
export USE_ABSTRACT=$USE_ABSTRACT
export USE_NODE_VECTOR=$USE_NODE_VECTOR
export TOP_N=$TOP_N
export HIDDEN_DIM=$HIDDEN_DIM
export BERT_VOCAB=/home/khenner/scibert_scivocab_uncased/vocab.txt
export BERT_WEIGHTS=/home/khenner/scibert_scivocab_uncased/weights.tar.gz
export TRAINING_DATA=$TRAINING_DATA
export DEV_DATA=$DEV_DATA
export TEST_DATA=$TEST_DATA

SERIALIZATION_DIR=/shared/2/projects/concite/serialization/"$LABEL_FIELD"_serialization/"$SEED"/model

if [ "$USE_ABSTRACT" == "true" ]; then
    SERIALIZATION_DIR="$SERIALIZATION_DIR"_BERT
    (( INPUT_DIM = INPUT_DIM + BERT_DIM ))
fi

if [ "$USE_NODE_VECTOR" == "true" ]; then
  SERIALIZATION_DIR="$SERIALIZATION_DIR"_n2v_"$EMB"
  (( INPUT_DIM = INPUT_DIM + EMBEDDING_DIM ))
  export PRETRAINED_FILE="$DATA_ROOT"data/embeddings/"$EMB".emb
else
  export PRETRAINED_FILE=None
fi

if [[ "$USE_NODE_VECTOR" == "false" && "$USE_ABSTRACT" == "false" ]]; then
  (( EMBEDDING_DIM = BERT_DIM + EMBEDDING_DIM ))
  (( INPUT_DIM = INPUT_DIM + EMBEDDING_DIM ))
fi

export INPUT_DIM=$INPUT_DIM
export EMBEDDING_DIM=$EMBEDDING_DIM

echo Serialization DIR: "$SERIALIZATION_DIR"
echo Vector FIle: "$PRETRAINED_FILE"
allennlp train allennlp_configs/acl_classifier.json -s $SERIALIZATION_DIR -f --include-package concite
