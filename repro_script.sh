#!/bin/bash

# 1. Install BanditPAM from the source
#pip install -r requirements.txt
#pip install -e .

# 2. Install datasets if necessary
cd data
# MNIST
if [ -e "MNIST_70k.csv" ]; then
    echo "MNIST found"
else
    echo "Installing MNIST..."
    curl -XGET https://motiwari.com/banditpam_data/MNIST_70k.tar.gz > MNIST_70k.tar.gz
    tar -xzvf MNIST_70k.tar.gz
    rm MNIST_70k.tar.gz
fi

# CIFAR-10
if [ -e "cifar10.csv" ]; then
    echo "CIFAR-10 found"
else
    echo "Installing CIFAR-10..."
    curl -XGET https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz > cifar-10-python.tar.gz
    tar -xzvf cifar-10-python.tar.gz
    rm cifar-10-python.tar.gz
    # Preprocess the dataset
    echo "Preprocessing the CIFAR dataset..."
    python preprocess_cifar.py
fi
cd -  # Go back to directory from where script was run

# 3. Run the experiments
python experiments/run_scaling_experiment.py
