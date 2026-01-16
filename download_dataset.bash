#!/usr/bin/env bash

echo Downloading the Dataset in Current working directory:
mkdir $PWD/dataset
curl -L -o $PWD/dataset/asl-alphabet.zip\
  https://www.kaggle.com/api/v1/datasets/download/grassknoted/asl-alphabet
echo Unziping...
unzip $PWD/dataset/asl-alphabet.zip -d $PWD/dataset
