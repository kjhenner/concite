HIDDEN_DIM=$1
USE_ABSTRACT=$2
USE_NODE_VECTOR=$3
EMB=$4

EMBEDDING_DIM=384
BERT_DIM=768
INPUT_DIM=0

DATA_ROOT="/shared/1/projects/concite/"
TEXT_LOOKUP_PATH="/shared/1/projects/concite/data/arc-paper-ids.tsv"

TRAINING_DATA="$DATA_ROOT"data/acl_data/train_sentence_co-occurrence.txt
DEV_DATA="$DATA_ROOT"data/acl_data/dev_sentence_co-occurrence.txt
TEST_DATA="$DATA_ROOT"data/acl_data/test_sentence_co-occurrence.txt

echo $TRAINING_DATA

export USE_ABSTRACT=$USE_ABSTRACT
export USE_NODE_VECTOR=$USE_NODE_VECTOR
export HIDDEN_DIM=$HIDDEN_DIM
export BERT_VOCAB=/home/khenner/scibert_scivocab_uncased/vocab.txt
export BERT_WEIGHTS=/home/khenner/scibert_scivocab_uncased/weights.tar.gz
export TRAINING_DATA=$TRAINING_DATA
export DEV_DATA=$DEV_DATA
export TEST_DATA=$TEST_DATA
export TEXT_LOOKUP_PATH=$TEXT_LOOKUP_PATH

SERIALIZATION_DIR=/shared/0/projects/concite/serialization/cocitation_serialization/"$SEED"/model

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
allennlp train allennlp_configs/cocitation_model.json -s $SERIALIZATION_DIR -f --include-package concite
