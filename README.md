NIRVAR
==============================

Network Informed Restricted Vector Autoregression

This repository contains the code and data used to obtain simulation study and applications results for 
"NIRVAR: Network Informed Restricted Vector Autoregression". The arXiv preprint is available at [https://www.arxiv.org/pdf/2407.13314](https://www.arxiv.org/pdf/2407.13314). 

Note that the financial returns data is too large to store on GitHub. The data is available upon request from b.martin22@imperial.ac.uk.

Installation
------------
You can install from pypi.org using 
`pip install nirvar` 

Alternatively, you can clone the repository using SSH or HTTPS:

`git clone git@github.com:bmartin9/NIRVAR.git` 

or 

`git clone https://github.com/bmartin9/NIRVAR.git`

Once cloned, change to the project root directory and install the nirvar package in edit mode using 

`pip install -e .` 

The package dependencies are listed in `environment.yaml` and can be installed using conda:

`conda env create --name envname -f environment.yaml` 

Usage
------------
If you have installed using pip, you can import classes and functions using, for example 

`from nirvar.models import train_model` 

If you have cloned the repository from GitHub and installed it in editable mode, use `src` instead of `nirvar`. For example,

`from src.models import train_model`

Example Notebook
----------------
`./notebooks/NIRVAR_example_usage.ipynb` is a Jupyter notebook that gives an example of how to simulate from the NIRVAR model and do NIRVAR estimation. 


Project Organization
------------

    ├── LICENSE            <- MIT
    ├── Makefile           <- Makefile based on cookiecutter data-science template
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── generated      <- Data generated from simulation studies as well as the predictions of NIRVAR/FARM/FNETS/GNAR on each application
    │   ├── processed      <- Transformed data used for model training
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Scripts for training NIRVAR/FARM/FNETS/GNAR models on application datasets.
                              Also contains scripts for NIRVAR simulation studies.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to transform data for applications 
    │       └── clean_stocks.py
    │       └── transform_raw_data.R
    │   │
    │   ├── models         <- Scripts to generate simulation data, train a NIRVAR model on data, and do predictions 
    │   │                     using trained model
    │   │   ├── generativeVAR.py
    │   │   └── train_model.py
        |   └── predict_model.py
    │   │
    │   └── visualization  <- Utility functions used to visualize results 
    │       └── utility_funcs.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


-------- 

Constructing Datasets
------------
The dataset used for the application to US industrial production was created using the August 2022 vintage of FRED-MD data, available at [https://research.stlouisfed.org/econ/mccracken/fred-databases/](https://research.stlouisfed.org/econ/mccracken/fred-databases/). The recommended fbi package was used to trasform the raw data. The fbi package is availabel at [https://github.com/cykbennie/fbi](https://github.com/cykbennie/fbi). The script used to implement the transformations for this project is `./src/data/transform_raw_data.R`. 
