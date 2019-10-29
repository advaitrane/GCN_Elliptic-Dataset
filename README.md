# GCN_Elliptic-Dataset
Graph Convolutional Network on data from Elliptic bitcoin dataset of transactions graph

# Instructions
## Requirements
- Python
- PyTorch
- pandas
- scikit-learn

## Dataset
- Download the [Elliptic Dataset](https://www.kaggle.com/ellipticco/elliptic-data-set#elliptic_bitcoin_dataset.zip) for Bitcoin transactions
- Don't change the names of the csv files
- If you change the name of the folder pass the path with the changed name as the command line argument for the dataset directory while training and testing


## Code Files
The files present in Code are: 
- util.py: Contains the function for loading the data
- model.py: Implementation of the Graph Convolutional Network with 2 layers
- train_GCN.py: Python script to train the GCN
- test_GCN.py: Python script to test the GCN using weights from training
- train_SkipGCN.py: Python script to train the SkipGCN
- test_SkipGCN.py: Python script to test the SkipGCN using weights from training
- GCN_Elliptic Dataset.ipynb: The main ipynb notebook used for all the tasks  

The weights for the models are provided in a folder titled gcn_weights. They can be obtained [here](https://drive.google.com/drive/folders/1b6ULpBjYsww0m9NHGjt5BMaRO0O0CbJa?usp=sharing)

## Usage
### Training the GCN
```
python train_GCN.py -d [:dataset directory path] -e [:number of epochs] -l [:learning rate] -t [:number of timesteps to train] -m [:directory to save model weights]
```

### Testing the GCN
```
python test_GCN.py -d [:dataset directory path] -t [:timestep to start testing] -m [:model weights directory]
```

### Training the SkipGCN
```
python train_SkipGCN.py -d [:dataset directory path] -e [:number of epochs] -l [:learning rate] -hs [:hidden layer size] -t [:number of timesteps to train] -m [:directory to save model weights]
```
Training the SkipGCN might be unstable and lead to NANs in the output. The problem might be solved by changing the size of the hidden layer. Use the -hs argument to change the hidden layer size. Use the same hidden layer size for training and testing. The model weights provided are for a hidden layer of size 16.

### Testing the SkipGCN
```
python test_SkipGCN.py -d [:dataset directory path] -t [:timestep to start testing] -hs [:hidden layer size] -m [:model weights directory]
``` 
