# CECP
This is the source code for CECP, "Charge Prediction by Constitutive Elements Matching of Crimes", which has been accepted by IJCAI-2022.

<br>

# Requirements
The program has been tested in the following environment: 
* Ubuntu 18.04
* Python 3.7.6
* Pytorch 1.7.1
* thulac 0.2.1
* scikit-learn 0.22.1
* numpy 1.18.1

<br>

# Structure
```
|-- ckpt/                   // obtained checkpoints from training

|-- data                    // folder to store data
    |-- CEs/                // raw data of constitutive elements
    |-- criminal/           // raw data of criminal dataset
    |-- cail/               // raw data of cail dataset
    |-- processed_data/     // processed data, represented by wordID
    |-- README.md

|-- logs/                   // folder to store results and training logs

|-- model/                  // code for training and testing
    |-- config.py           // hyperparameters
    |-- customer_layers.py  // code for aggregation, PFI, and action
    |-- encoder.py          // code for encoder
    |-- main.py             // train and test
    |-- model.py            // code for environment, agent, and predictor
    |-- util.py             //

|-- utils                   // 
    |-- preprocess_data.py  // preprocessing
    |-- read_save_data.py   // processing data and saving them
```

<br>

# Quick Start
First, download the processed data from [processed data example - Baidu](https://pan.baidu.com/s/1pk8-h-UYGKfl31pMqmdsFA?pwd=itmd) or [processed data example - Google](https://drive.google.com/file/d/1I753whBt5yPHmE9z5wgQen2rNWdoazPY/view?usp=sharing) and unzip them to **./data/processed_data/**.

For training and testing:
```
cd model
python main.py --dataset criminal --nclass 149
```

