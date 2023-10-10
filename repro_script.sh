#!/bin/bash

# 1. Install the BanditPAM package

# Install prerequisites
pip install -r requirements.txt
brew install libomp armadillo

# Remove files possibly left over from previous builds
sudo rm -rf build && mkdir build && sudo rm -rf banditpam.cpython-* banditpam.egg-info banditpam.egg-info/ tmp/ build/ && sudo python -m pip uninstall -y banditpam

# Install BanditPAM package
sudo python -m pip install --no-use-pep517  --no-cache-dir --ignore-installed -vvvvv -e .

# 2. Install datasets if necessary
mkdir -p data
cd data

# MNIST
if [ -e "MNIST_70k.csv" ]; then
    echo "MNIST found"
else
    echo "Downloading MNIST..."
    curl -XGET https://motiwari.com/banditpam_data/MNIST_70k.tar.gz > MNIST_70k.tar.gz
    tar -xzvf MNIST_70k.tar.gz
    rm MNIST_70k.tar.gz
fi

# CIFAR-10
if [ -e "cifar10.csv" ]; then
    echo "CIFAR-10 found"
else
    echo "Downloading CIFAR-10..."
    curl -XGET https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz > cifar-10-python.tar.gz
    tar -xzvf cifar-10-python.tar.gz
    rm cifar-10-python.tar.gz
    # Preprocess the dataset
    echo "Preprocessing the CIFAR dataset..."
    python preprocess_cifar.py
fi

# scRNA
if [ -e "reduced_scrna.csv" ]; then
    echo "scRNA found"
else
    echo "Downloading scRNA..."
    curl -XGET https://motiwari.com/banditpam_data/scrna_reformat.csv.gz > scRNA_reformat.csv.gz
    # Preprocess the dataset. Don't need to unzip because pandas can read .csv.gz files
    echo "Preprocessing the scRNA dataset..."
    python preprocess_scrna.py
fi

# 20 Newsgroups
if [ -e "20_newsgroups.csv" ]; then
  echo "20 Newsgroups found"
else
  echo "Preprocessing 20 newsgroups"
  python newsgroups_to_csv.py
fi

cd -  # Go back to directory from where script was run

# 3. Run the experiments
python repro_script.py
