
USE_ABSTRACT=true
USE_NODE_VECTOR=true

INTENT_WT=0.5
EMB_TYPE=combined

ABSTRACT_SIZE=768
NODE_VECTOR_SIZE=384
EMBEDDING_DIM=384
COMBINED_SIZE=1152

DATA_ROOT="/shared-1/projects/concite/"

if [ "$USE_ABSTRACT" == "true" ] && [ "$USE_NODE_VECTOR" == "true" ]; then
    SERIALIZATION_DIR="$DATA_ROOT"/serialization/workshop_abstract_n2v_"$EMB_TYPE"_"$INTENT_WT"
elif [ "$USE_ABSTRACT" == "true" ]; then
    SERIALIZATION_DIR="$DATA_ROOT"/serialization/workshop_abstract
else
    SERIALIZATION_DIR="$DATA_ROOT"/serialization/workshop_n2v_"$EMB_TYPE"_"$INTENT_WT"
fi

export BERT_VOCAB=/home/khenner/scibert_scivocab_uncased/vocab.txt
export BERT_WEIGHTS=/home/khenner/scibert_scivocab_uncased/weights.tar.gz
export PRETRAINED_FILE="$DATA_ROOT"/data/"$EMB_TYPE"_40_384_0.5_0.5_"$INTENT_WT".emb
export EMBEDDING_DIM=$EMBEDDING_DIM

if [ "$USE_ABSTRACT" == true ] && [ "$USE_NODE_VECTOR" == true ]; then
    export INPUT_DIM=$COMBINED_SIZE
elif [ "$USE_ABSTRACT" == true ]; then
    export INPUT_DIM=$ABSTRACT_SIZE
else
    export INPUT_DIM=$EMBEDDING_DIM
fi

echo $SERIALIZATION_DIR
#allennlp train allennlp_configs/predict_acl_venue.json -s $SERIALIZATION_DIR -f --include-package concite
