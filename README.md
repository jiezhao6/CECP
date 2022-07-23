# CECP
The source code and Appendix of our work "Charge Prediction by Constitutive Elements Matching of Crimes", IJCAI 2022. [https://www.ijcai.org/proceedings/2022/627](https://www.ijcai.org/proceedings/2022/627)

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

<br>

# Reference
BibTex:
```
@inproceedings{ijcai2022-627,
  title     = {Charge Prediction by Constitutive Elements Matching of Crimes},
  author    = {Zhao, Jie and Guan, Ziyu and Xu, Cai and Zhao, Wei and Chen, Enze},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Lud De Raedt},
  pages     = {4517--4523},
  year      = {2022},
  month     = {7},
  note      = {Main Track}
  doi       = {10.24963/ijcai.2022/627},
  url       = {https://doi.org/10.24963/ijcai.2022/627},
}
```