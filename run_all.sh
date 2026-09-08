#!/usr/bin/env bash
# Runs every experiment and tees the console output into logs/.
set -e
mkdir -p logs
for s in 01_regression_diabetes_linear.py \
         02_regression_autompg_ridge.py \
         03_classification_iris_softmax.py \
         04_classification_breast_cancer_logistic.py \
         05_nn_fashion_mnist_mlp.py \
         06_nn_shakespeare_rnn.py \
         07_nn_cifar10_cnn.py \
         08_nn_imdb_embedding.py ; do
    echo ">>> $s"
    python "$s" 2>&1 | tee "logs/${s%.py}.log"
done
echo "All experiments finished. Figures are in ./figures, metrics in ./results."
