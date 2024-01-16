# Modelling of the Impact of Automated Income Fraud Detection

This repository contains the code for the thesis _Modelling of the Impact of Automated Income Fraud Detection: a Case Study in the Netherlands_

## Usage

### Setup.

#### Requirements

For clearer specification of our setup, we make use of
[Poetry](https://python-poetry.org/) to keep track of dependencies and python
versions. Details such as python and package versions can be found in the
generated [pyproject.toml](pyproject.toml) and [poetry.lock](poetry.lock) files.

For poetry users, getting setup is as easy as running

```terminal
poetry install
```

We also provide an [environment.yml](environment.yml) file for
[Conda](https://docs.conda.io/projects/conda/en/latest/index.html) users who do
not wish to use poetry. In this case simply run

```terminal
conda env create -f environment.yml
```

Finally, if neither of the above options are desired, we also provide a
[requirements.txt](requirements.txt) file for
[pip](https://pypi.org/project/pip/) users. In this case, simply run

```terminal
pip install -r requirements.txt
```
