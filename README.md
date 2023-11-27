# Multi-region-attention

To see the details of the model architecture, go to [Architecture](./docs/markdowns/architecture.md)
To see the underlying mechanism for Multiscale Attention, go to [Multiscale Attention](./docs/markdowns/multiscale_attention.md)



# File structure 
```
├── data
│   ├── ADE20K
│   ├── CityScapes
├── config
│   ├── config_ade.py
│   ├── config_cityscapes.py
├── dataloader
│   ├── ade_20k
│   ├── cityscapes
├── docs
├── models
│   ├── encoders
│   ├── decoders
│   ├── builders.py    <- Base model builder.
├── notebooks
│   ├── exploratory
│   ├── analysis
├── pretrained
├── results
├── utils
├── requirements.txt   <- The requirements file for reproducing the environment, e.g.
├── setup.py           <- Make this project pip installable with `pip install -e`
├── train.py
├── test.py
├── validation.py
├── multiscale_evaluation.py
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
```


# How to run

1. Clone the project using git clone https://github.com/mahfuzalhasan/multi-region-attention.git
2. Run setup.py to install all the required libraries
3. Modify the config file to change the model parameters and hyperparameters.
4. Run train.py with the specific dataset name (example: python train.py cityscapes) 

	
