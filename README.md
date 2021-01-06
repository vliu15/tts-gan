# End-to-end Text-to-Speech with Generative Adversarial Networks

This repository contains implementation and end-to-end training scripts for text-to-speech models, based off
[End-to-End Adversarial Text-to-Speech (Donahue etal. 2020)](https://arxiv.org/abs/2006.03575).

## Usage
To setup the Python environment, run

```bash
python -m venv ttsgan
source ttsgan/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Training is abstracted via the [Pytorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) package.

Aggregate audio files from the LJ-Speech dataset by running
```bash
ls LJSpeech-1.1/wavs/*.wav | tail -n+10 > train_files.txt
ls LJSpeech-1.1/wavs/*.wav | head -n10 > test_files.txt
```

Download the CMU Dict phonemizer [here](http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/sphinxdict/cmudict_SPHINX_40) and edit the fields at the top of `config.yml` to point to the corresponding files on your system.

To train, simply run
```bash
python train.py -c config.yml
```
