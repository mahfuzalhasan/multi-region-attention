# multi-region-attention






# File structure 

├── LICENSE
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── ADE20K       
│   ├── CityScapes        
│
├── docs               
│
├── models
│   ├── Pretrained       
│   ├── Saved           
│
├── notebooks
│   ├── Exploratory       
│   ├── Analysis
│
├── clusters
│   ├── Nautilous       
│   ├── Slarm
│  
├── output            <- Generated image output
│
├── requirements.txt   <- The requirements file for reproducing the environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- Make this project pip installable with `pip install -e`
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── make_dataset.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   └── train_model.py
