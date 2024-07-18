# Clothes classification with Fashion-MNIST

[Fashion-MNIST](https://github.com/cansuyildiim/fashion-mnist), Fashion-MNIST is a dataset for clothing classification. It contains 70,000 images from 10 different classes, divided into 60,000 images for training and 10,000 images for testing. Deep learning models achieve an accuracy of approximately 95%, which is the official benchmark. We aim to train models that achieve similar accuracy and are ‘efficient’ to use on a regular computer.
In this exploratory study, we have three different goals:
Train different models using various model architectures and training strategies.
Discuss the results of different experiments.
Test the trained models using a demo with a webcam.

## Launch

We recommend using an isolated Python environment with at least Python 3.6, such as venv or conda. Then, you can set it up using the following code:

```bash
git clone https://github.com/cansuyildiriim/Fashion-MNIST-PyTorch.git
cd Fashion-MNIST-PyTorch
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

### PyTorch Installation

Since the installation of [PyTorch](https://pytorch.org/get-started/locally/) varies for each platform, please refer to the PyTorch installation guide.

## Usage

After completing the setup, use the `train_fashionMNIST.py` script to replicate the results of different experiments.

### Project Structure

The project contains five different folders:
data: This directory is created when the train_fashionMNIST.py script is run for the first time. It contains the training and test datasets of Fashion MNIST.
demo: This directory contains all the code for the demo.
experiments: This directory is created when the train_fashionMNIST.py script is run for the first time. It contains the results of the experiments. Placing trained models in this directory allows you to use them with the demo.
images: This directory contains the images used in this README file.
models: This directory contains the architecture of the models and the definition of the labels.

- **data:** This directory is created when the train_fashionMNIST.py script is run for the first time. It contains the training and test datasets of Fashion MNIST.
- **demo:** This directory contains all the code for the demo.
- **experiments: This directory is created when the train_fashionMNIST.py script is run for the first time. It contains the results of the experiments. Placing trained models in this directory allows you to use them with the demo.
- **images:** This directory contains the images used in this README file.
- **models:** This directory contains the architecture of the models and the definition of the labels

## Demo

In this project, there is a demo application for real-time image processing. To run the demo application, a Python script named `run_inference.py` is used. This script performs image classification using trained models and displays the results on the screen.

Use the following commands to run the demo:

```bash
cd demo
python3 -m venv .env
source .env/bin/activate
pip install -r requirements
```

```bash
cd demo
python run_inference.py --model SimpleCNNModel --weights_path ../demo/experiments/01_SimpleCNNModel/model_best.pth.tar --display_input --display_fps --device 0
```
