USE_ABSTRACT=$1
USE_NODE_VECTOR=$2

EMB_TYPE=$3 # "all" or "combined"
INTENT_WT=$4

TOP_N_WORKSHOPS=$5

ABSTRACT_SIZE=768
NODE_VECTOR_SIZE=384
EMBEDDING_DIM=384
COMBINED_SIZE=1152

DATA_ROOT="/shared-1/projects/concite/"

export USE_ABSTRACT=$USE_ABSTRACT
export USE_NODE_VECTOR=$USE_NODE_VECTOR
export TOP_N_WORKSHOPS=$TOP_N_WORKSHOPS

if [ "$USE_ABSTRACT" == "true" ] && [ "$USE_NODE_VECTOR" == "true" ]; then
    SERIALIZATION_DIR="$DATA_ROOT"/serialization/"$TOP_N_WORKSHOPS"_workshop_abstract_n2v_"$EMB_TYPE"
elif [ "$USE_ABSTRACT" == "true" ]; then
    SERIALIZATION_DIR="$DATA_ROOT"/serialization/"$TOP_N_WORKSHOPS"_workshop_abstract
else
  if [ "$EMB_TYPE" == "combined" ]; then
    SERIALIZATION_DIR="$DATA_ROOT"/serialization/"$TOP_N_WORKSHOPS"_workshop_n2v_"$EMB_TYPE"
  else
    SERIALIZATION_DIR="$DATA_ROOT"/serialization/"$TOP_N_WORKSHOPS"_workshop_n2v_"$EMB_TYPE"
  fi
fi

if [ "$EMB_TYPE" == "combined" ]; then
  SERIALIZATION_DIR="$SERIALIZATION_DIR"_"$INTENT_WT"
fi

export BERT_VOCAB=/home/khenner/scibert_scivocab_uncased/vocab.txt
export BERT_WEIGHTS=/home/khenner/scibert_scivocab_uncased/weights.tar.gz

if [ "$EMB_TYPE" == "combined" ]; then
  export PRETRAINED_FILE="$DATA_ROOT"/data/"$EMB_TYPE"_40_384_0.5_0.5_"$INTENT_WT".emb
elif [ "$EMB_TYPE" == "all" ]; then
  export PRETRAINED_FILE="$DATA_ROOT"/data/"$EMB_TYPE"_40_384_0.5_0.5.emb
else
  export PRETRAINED_FILE=None
fi

export EMBEDDING_DIM=$EMBEDDING_DIM

if [ "$USE_ABSTRACT" == "true" ] && [ "$USE_NODE_VECTOR" == "true" ]; then
    export INPUT_DIM=$COMBINED_SIZE
elif [ "$USE_ABSTRACT" == "true" ]; then
    export INPUT_DIM=$ABSTRACT_SIZE
else
    export INPUT_DIM=$EMBEDDING_DIM
fi

echo Serialization DIR: "$SERIALIZATION_DIR"
echo Vector FIle: "$PRETRAINED_FILE"
echo $INPUT_DIM
allennlp train allennlp_configs/predict_acl_venue.json -s $SERIALIZATION_DIR -f --include-package concite
