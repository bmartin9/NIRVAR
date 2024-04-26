NIRVAR
==============================

Network Informed Restricted Vector Autoregression

This repository contains the code and data used to obtain simulation study and applications results for 
the NIRVAR paper.

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


Project Organization
------------

    ├── LICENSE            <- MIT
    ├── Makefile           <- Makefile based on cookiecutter data-science template
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── generated      <- Data generated from simulation studies
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
    │   └── visualization  <- Scripts to visualize results 
    │       └── 0.3-ARI-comparisons.py
    │       └── 0.3-embedding-dim.py
    │       └── 0.3-SICCD-bars-plot.py
    │       └── 0.3-turnover.py
    │       └── 0.3-visualise-backtesting.py
    │       └── factors_over_time.py
    │       └── utility_funcs.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

