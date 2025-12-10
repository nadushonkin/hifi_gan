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

0. Open Google Colab notebook (yours or `demo.ipynb` from this repo)

1. Clone this repo
```bash
!git clone https://github.com/nadushonkin/hifi_gan
```

2. Create and activate new environment in Colab
```bash
!pip install -q condacolab
import condacolab
condacolab.install()
```

```bash
!conda create -n hifi_env python=3.9 -y -q
!source /usr/local/etc/profile.d/conda.sh && conda activate hifi_env
```

3. Install all required packages

   ```bash
   !pip install -r /content/hifi_gan/requirements.txt --quiet
   ```


## How To Use

To download model checkpoint:
   ```bash
   import gdown
   import os

   url = "https://drive.google.com/file/d/1pbIsOneIR0LoH85KhxYum98wM2jiZKDv/view?usp=sharing"

   output = "model_weights.pth"

   gdown.download(url, output, fuzzy=True)
   ```

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments. Note that you need to pass the link to you dataset for training (use arguments datasets.train.data_dir="path_to_your_dataset" and datasets.test.data_dir="path_to_your_dataset")
If you want to continue training from checkpoint, you can use trainer.resume_from="path to your checkpoint".
If you want to see info in CometML, modify writer config and don't forget to add `COMET_API_KEY="your_API_key"` to your command

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
You can also look at `demo.ipynb` in this repo for more details.

## Reports
You can find the logs for the training of my model from the start of the training [here](https://www.comet.com/nadushonkin/dla-nv/kxrj4xsf3yvzfwxmq4v9kqk0uoywsp6t?compareXAxis=step&experiment-tab=panels&showOutliers=true&smoothing=0&xAxis=step)

Also you can find the **report** [here](https://www.comet.com/nadushonkin/dlanv/reports/NoNUGpvOapUMxpyVxGQCeiMFK)


## Credits

This repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) and [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repositories.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
