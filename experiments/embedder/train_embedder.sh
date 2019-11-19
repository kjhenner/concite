TEXT_FIELD=$1
USE_TEXT=$2
USE_NODE_VECTORS=$3

EMB_TYPE=$4 # "all" or "combined"
EMB_L=20
EMB_P=0.3
EMB_Q=0.7


EMBEDDING_DIM=384
BERT_DIM=768
INPUT_DIM=0

DATA_ROOT="/shared-1/projects/concite/"

TRAINING_DATA="$DATA_ROOT"data/acl_data/train_section_co-occurrence.txt
DEV_DATA="$DATA_ROOT"data/acl_data/dev_section_co-occurrence.txt
TEST_DATA="$DATA_ROOT"data/acl_data/test_section_co-occurrence.txt

PAPER_LOOKUP_PATH="$DATA_ROOT"data/acl_data/acl_data.jsonl

echo $TRAINING_DATA

export TEXT_FIELD=$TEXT_FIELD
export USE_TEXT=$USE_TEXT
export USE_NODE_VECTORS=$USE_NODE_VECTORS
export BERT_VOCAB=/home/khenner/scibert_scivocab_uncased/vocab.txt
export BERT_WEIGHTS=/home/khenner/scibert_scivocab_uncased/weights.tar.gz
export TRAINING_DATA=$TRAINING_DATA
export DEV_DATA=$DEV_DATA
export TEST_DATA=$TEST_DATA
export PAPER_LOOKUP_PATH=$PAPER_LOOKUP_PATH

EMB_SUFFIX="$EMB_TYPE"_"$EMB_L"_"$EMBEDDING_DIM"_"$EMB_P"_"$EMB_Q"
if [ "$EMB_TYPE" == "combined" ]; then
    INTENT_WT=0.5
    EMB_SUFFIX="$EMB_SUFFIX"_"$INTENT_WT"
fi

SERIALIZATION_DIR="$DATA_ROOT"/embedder_serialization/"$TEXT_FIELD"

if [ "$USE_TEXT" == "true" ]; then
    SERIALIZATION_DIR="$SERIALIZATION_DIR"_bert
    (( INPUT_DIM = INPUT_DIM + BERT_DIM ))
fi

if [ "$USE_NODE_VECTORS" == "true" ]; then
    SERIALIZATION_DIR="$SERIALIZATION_DIR"_n2v_"$EMB_TYPE"
    (( INPUT_DIM = INPUT_DIM + EMBEDDING_DIM ))
    export PRETRAINED_FILE="$DATA_ROOT"data/embeddings/"$EMB_SUFFIX".emb
else
    export PRETRAINED_FILE=None
    (( EMBEDDING_DIM = BERT_DIM + EMBEDDING_DIM ))
fi

if [ "$EMBEDDED_TEXT" == "title" ]; then
  export TEXT_LOOKUP_PATH="$DATA_ROOT"data/arc-paper-ids.tsv
else
  export TEXT_LOOKUP_PATH="$DATA_ROOT"data/acl_data/acl_data.jsonl
fi

export INPUT_DIM=$INPUT_DIM
export EMBEDDING_DIM=$EMBEDDING_DIM

echo Serialization DIR: "$SERIALIZATION_DIR"
echo Vector FIle: "$PRETRAINED_FILE"
allennlp train allennlp_configs/acl_embedder.json -s $SERIALIZATION_DIR -f --include-package concite
rm "$SERIALIZATION_DIR"/training_state_epoch_*
rm "$SERIALIZATION_DIR"/model_state_epoch_*
