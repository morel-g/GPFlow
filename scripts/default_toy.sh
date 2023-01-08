#!/bin/bash

DIR="$( cd "$( dirname "$0" )" && pwd )"
NF_MODEL='bnaf'
OPT_TYPE="train_map"
DATA_TYPE='moons'
GPU=0
USE_EULER=false
EULER_CASE="penalization"

if $USE_EULER 
then
    python $DIR/../main.py --default_params --data_type $DATA_TYPE --nf_model $NF_MODEL --opt_type $OPT_TYPE --default_model_path --gpu $GPU --use_euler $USE_EULER --euler_case $EULER_CASE
else
    python $DIR/../main.py --default_params --data_type $DATA_TYPE --nf_model $NF_MODEL --opt_type $OPT_TYPE --default_model_path --gpu $GPU
fi