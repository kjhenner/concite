HIDDEN_DIM=$1
TOP_N_WORKSHOPS=$2
USE_ABSTRACT=$3
USE_NODE_VECTOR=$4

EMB_TYPE=$5 # "all" or "combined"
EMB_L=$6
EMB_P=$7
EMB_Q=$8
INTENT_WT=$9

echo $HIDDEN_DIM

EMBEDDING_DIM=384
COMBINED_SIZE=1152

DATA_ROOT="/shared-1/projects/concite/"

SERIALIZATION_DIR_NAME=serialization_"$HIDDEN_DIM"

export USE_ABSTRACT=$USE_ABSTRACT
export USE_NODE_VECTOR=$USE_NODE_VECTOR
export TOP_N_WORKSHOPS=$TOP_N_WORKSHOPS
export HIDDEN_DIM=$HIDDEN_DIM

EMB_SUFFIX="$EMB_TYPE"_"$EMB_L"_"$EMBEDDING_DIM"_"$EMB_P"_"$EMB_Q"
if [ "$EMB_TYPE" == "combined" ]; then
  EMB_SUFFIX="$EMB_SUFFIX"_"$INTENT_WT"
fi

if [ "$USE_ABSTRACT" == "true" ] && [ "$USE_NODE_VECTOR" == "true" ]; then
    SERIALIZATION_DIR="$DATA_ROOT"/"$SERIALIZATION_DIR_NAME"/"$TOP_N_WORKSHOPS"_workshop_abstract_n2v_"$EMB_SUFFIX"
elif [ "$USE_ABSTRACT" == "true" ]; then
    SERIALIZATION_DIR="$DATA_ROOT"/"$SERIALIZATION_DIR_NAME"/"$TOP_N_WORKSHOPS"_workshop_abstract
else
  if [ "$EMB_TYPE" == "combined" ]; then
    SERIALIZATION_DIR="$DATA_ROOT"/"$SERIALIZATION_DIR_NAME"/"$TOP_N_WORKSHOPS"_workshop_n2v_"$EMB_SUFFIX"
  else
    SERIALIZATION_DIR="$DATA_ROOT"/"$SERIALIZATION_DIR_NAME"/"$TOP_N_WORKSHOPS"_workshop_n2v_"$EMB_SUFFIX"
  fi
fi


export BERT_VOCAB=/home/khenner/scibert_scivocab_uncased/vocab.txt
export BERT_WEIGHTS=/home/khenner/scibert_scivocab_uncased/weights.tar.gz


if [ "$USE_NODE_VECTOR" == "true" ]; then
  export PRETRAINED_FILE="$DATA_ROOT"/data/"$EMB_SUFFIX".emb
else
  export PRETRAINED_FILE=None
fi

export EMBEDDING_DIM=$EMBEDDING_DIM
export INPUT_DIM=$COMBINED_SIZE

echo Serialization DIR: "$SERIALIZATION_DIR"
echo Vector FIle: "$PRETRAINED_FILE"
allennlp train allennlp_configs/predict_acl_venue.json -s $SERIALIZATION_DIR -f --include-package concite
rm "$SERIALIZATION_DIR"/training_state_epoch_*
rm "$SERIALIZATION_DIR"/model_state_epoch_*
