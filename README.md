# multi-region-attention


[Architecture](./docs/markdowns/architecture.md)
[Multiscale Attention](./docs/markdowns/multiscale_attention.md)



# File structure 
```
├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── ADE20K
│   ├── CityScapes
│
├── docs
│
├── pretrained
├── notebooks
│   ├── exploratory
│   ├── analysis
│
├── clusters
│   ├── nautilus
│   ├── slurm
│
├── output             <- Generated image output
├── results            <- Trained model, log
├── requirements.txt   <- The requirements file for reproducing the environment, e.g.
│
├── setup.py           <- Make this project pip installable with `pip install -e`
├── models
│   │
│   ├── encoders
│   ├── decoders
│   ├── builders.py    <- Base model builder. 
├── dataloader
│   ├── ade_20k
│   ├── cityscapes
├── config
│   ├── config_ade.py
│   ├── config_cityscapes.py
├── utils
├── train.py
├── test.py
├── validation.py
├── multiscale_evaluation.py
```


	
