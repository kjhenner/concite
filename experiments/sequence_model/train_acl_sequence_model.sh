EMBEDDED_TEXT=$1
HIDDEN_DIM=$2
USE_ABSTRACTS=$3
USE_NODE_VECTORS=$4

EMB_TYPE=$5 # "all" or "combined"
EMB_L=$6
EMB_P=$7
EMB_Q=$8
INTENT_WT=${9}

EMBEDDING_DIM=384
BERT_DIM=768
INPUT_DIM=0

DATA_ROOT="/shared-1/projects/concite/"

TRAINING_DATA="$DATA_ROOT"data/acl_data/train_acl_citation_sequences.txt
DEV_DATA="$DATA_ROOT"data/acl_data/dev_acl_citation_sequences.txt
TEST_DATA="$DATA_ROOT"data/acl_data/test_acl_citation_sequences.txt

echo $TRAINING_DATA

export CONTEXTUALIZER=lstm
export EMBEDDED_TEXT=$EMBEDDED_TEXT
export USE_ABSTRACTS=$USE_ABSTRACTS
export USE_NODE_VECTORS=$USE_NODE_VECTORS
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

SERIALIZATION_DIR="$DATA_ROOT"/sequence_serialization_"$HIDDEN_DIM"/citations

echo $USE_ABSTRACTS
if [ "$USE_ABSTRACTS" == "true" ]; then
    SERIALIZATION_DIR="$SERIALIZATION_DIR"_abstract
    (( INPUT_DIM = INPUT_DIM + BERT_DIM ))
fi

echo $USE_ABSTRACTS
if [ "$USE_NODE_VECTORS" == "true" ]; then
    SERIALIZATION_DIR="$SERIALIZATION_DIR"_n2v_"$EMB_SUFFIX"
    (( INPUT_DIM = INPUT_DIM + EMBEDDING_DIM ))
    export PRETRAINED_FILE="$DATA_ROOT"data/embeddings/"$EMB_SUFFIX".emb
else
    export PRETRAINED_FILE=None
fi

if [ "$EMBEDDED_TEXT" == "title" ]; then
  export TEXT_LOOKUP_PATH="$DATA_ROOT"data/arc-paper-ids.tsv
else
  export TEXT_LOOKUP_PATH="$DATA_ROOT"data/acl_data/acl_data.jsonl
fi

export INPUT_DIM=$INPUT_DIM

echo Serialization DIR: "$SERIALIZATION_DIR"
echo Vector FIle: "$PRETRAINED_FILE"
echo $INPUT_DIM
allennlp train allennlp_configs/acl_sequence_model.json -s $SERIALIZATION_DIR -f --include-package concite
#rm "$SERIALIZATION_DIR"/training_state_epoch_*
#rm "$SERIALIZATION_DIR"/model_state_epoch_*
