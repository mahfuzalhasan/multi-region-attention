# Multi-region-attention

Our model architecture is illustrated and briefly described -->  [Architecture](./docs/markdowns/architecture.md)

A closer look at Multiscale Region Attention --> [Multiscale Attention](./docs/markdowns/multiscale_attention.md)



### Setup

1. Clone the project using *git*
    ```
    git clone https://github.com/mahfuzalhasan/multi-region-attention.git
    ``````
2. Run *setup.py* to install all the required libraries



### Training
3. Browse the config files (~/configs/*) to tune the dataset-specific hyperparameters and system-specific settings (eg no of GPUs to use).
4. Run train.py followed by the name of a specific dataset (example: python train.py cityscapes)
    ```
    python train.py <dataset-name>
    ```
    eg: For training on Cityscapes:
    ```
 	python train.py cityscapes
    ```
 	
    For training on ADE20k:
    ```
 	python train.py ade
    ```

### Validation
5. Include desired model path in *multiscale_evaluation.py* and run:
    ```
    python multiscale_evaluation.py
    ``` 



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

---