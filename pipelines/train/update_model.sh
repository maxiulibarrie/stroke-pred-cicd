#!/bin/bash

dvc remove $MODEL_DVC

dvc add $MODEL_PATH --to-remote -r $MODEL_TRACK_NAME

dvc push $MODEL_DVC -r $MODEL_TRACK_NAME
