#!/usr/bin/env bash

# run this script on the root
mkdir -p data/moving_mnist
cd data/moving_mnist

# download mmnist and place them in `data/moving_mnist/`
wget -c http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy
wget -c -O train-images-idx3-ubyte.gz https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
wget -c -O train-labels-idx1-ubyte.gz https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz

# regenerate the canonical fixed split labels alongside the fixed test sequence
cd ../../
python tools/prepare_data/generate_mmnist.py mnist
python tools/prepare_data/generate_mmnist.py mnist_cifar
cd data/moving_mnist

# download the test set of mmnist_cifar
wget https://github.com/chengtan9907/OpenSTL/releases/download/v0.1.0/mnist_cifar_test_seq.npy.tar
tar -xzvf mnist_cifar_test_seq.npy.tar

echo "finished"


# Download and arrange them in the following structure:
# OpenSTL
# └── data
#     ├── moving_mnist
#     │   ├── mnist_cifar_test_seq.npy
#     │   ├── mnist_test_seq.npy
#     │   ├── mnist_test_seq_labels.npy
#     │   ├── train-images-idx3-ubyte.gz
#     │   ├── train-labels-idx1-ubyte.gz
