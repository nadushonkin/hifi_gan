# HIFI-GAN

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

<p align="center">
<a href="https://github.com/Blinorot/pytorch_project_template/generate">
  <img src="https://img.shields.io/badge/use%20this-template-green?logo=github">
</a>
<a href="https://github.com/Blinorot/pytorch_project_template/blob/main/LICENSE">
   <img src=https://img.shields.io/badge/license-MIT-blue.svg>
</a>
<a href="https://github.com/Blinorot/pytorch_project_template/blob/main/CITATION.cff">
   <img src="https://img.shields.io/badge/cite-this%20repo-purple">
</a>
</p>

## About

This repository contains implementation of [HIFI-GAN](https://arxiv.org/pdf/2010.05646)

## Installation

Installation may depend on your task. The general steps are the following:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

## How To Use

To download model checkpoint:
   ```bash
   import gdown
   import os

   url = "https://drive.google.com/file/d/1ibjHvTtlUEMTqVeecTmbjcXOxEr-rYXs/view?usp=sharing"

   output = "model_weights.pth"

   gdown.download(url, output, fuzzy=True)
   ```

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

To generate audio from dataset (evaluate the model or save predictions):

```bash
python synthesize.py \
                  inferencer.from_pretrained="path_to_checkpoint_weights" \
                  dataloader.batch_size=1 \
                  datasets.test.data_dir="dataset_path" \
                  datasets.test.resynthesize=True \
                  inferencer.log=False
```
To generate audio from command line input (evaluate the model or save predictions):
```bash
python synthesize.py \
                  --config-name=query \
                  inferencer.from_pretrained="path_to_checkpoint_weights" \
                  query="'type your query here'" \
                  inferencer.log=False
```

If you want to see info in CometML, use these:
```bash
COMET_API_KEY="your_API_key" \
                  python synthesize.py \
                  inferencer.from_pretrained="path_to_checkpoint_weights" \
                  dataloader.batch_size=1 \
                  datasets.test.data_dir="dataset_path" \
                  datasets.test.resynthesize=True
```
You can also look at demo.ipynb in this repo for more details.


## Credits

This repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) and [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repositories.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
