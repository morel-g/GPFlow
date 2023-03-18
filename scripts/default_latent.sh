#!/bin/bash

DIR="$( cd "$( dirname "$0" )" && pwd )"
NF_MODEL='cpflow'
OPT_TYPE="train_nf"
DATA_TYPE='mnist'
GPU=2
USE_EULER=false

if $USE_EULER 
then
    python $DIR/../main.py --default_params --nf_model $NF_MODEL --data_type $DATA_TYPE --opt_type $OPT_TYPE --gpu $GPU --use_euler
else
    python $DIR/../main.py --default_params --nf_model $NF_MODEL --data_type $DATA_TYPE --opt_type $OPT_TYPE --gpu $GPU
fi

