NIRVAR
==============================

Network Informed Restricted Vector Autoregression

Project Organization
------------

    ├── LICENSE            <- MIT
    ├── Makefile           <- Makefile based on cookiecutter data-science template
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Scripts for training NIRVAR/FARM/FNETS/GNAR models on application datasets.
                              Also contains scripts for NIRVAR simulation studies.
    │
    ├── notebooks          <- Jupyter notebooks.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Link to NIRVAR paper
    │   └── figures        <- Generated graphics and figures used in NIRVAR paper
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── 
    │   │
    │   ├── models         <- Scripts to generate simulation data, train a NIRVAR model on data, and do predictions 
    │   │                     using trained model
    │   │   ├── generativeVAR.py
    │   │   └── train_model.py
        |   └── predict_model.py
    │   │
    │   └── visualization  <- Scripts to visualize results 
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

