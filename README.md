# Diffusion Simulator Generator
Python toolbox for generating simulated diffusion MRI data using biophysical or signal models. Uses [DMIPY](https://github.com/AthenaEPI/dmipy) and [DIPY](https://dipy.org/) for building models. See the `example_diffsimgen.ipynb` file for an example of how to use the toolbox.

## Installation
Dependency management and python package handling is done with [poetry](https://python-poetry.org/docs/).
```
pip install poetry
```
then the toolbox can be installed with:
```
git clone https://github.com/Bradley-Karat/DiffSimGen.git
cd DiffSimGen
poetry install
```
