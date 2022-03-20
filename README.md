# ALTI in NMT

We extend ALTI method presented in [Ferrando et al., 2022](https://arxiv.org/abs/2203.04212) to the encoder-decoder setting.
## Usage with M2M model

We follow fairseq [M2M](https://github.com/pytorch/fairseq/tree/main/examples/m2m_100) instructions and download the models (412M and 1.2B), dictionary, and SPM tokenizer in `M2M_CKPT_DIR`. Then, set the environmental variable:

```bash
export M2M_CKPT_DIR=... # Path to checkpoint,dictionary, and SPM
````

`m2m_interpretability.ipynb` can be run in generate or interactive mode.

### To use it in interactive mode, tokenize your own test data using:
```bash
python /path/to/fairseq/scripts/spm_encode.py \
    --model spm.128k.model \
    --output_format=piece \
    --inputs=/path/to/input/file/here \
    --outputs=/path/to/output/file/here
```

and select `data_sample = 'interactive'` in `m2m_interpretability.ipynb`.

<mark>Already tokenized data for testing the notebook (de-en) is available in `./data`.</mark>

### To use it in generate mode, provide the path to the binarized data by setting the environment variable `M2M_DATA_DIR`:
```bash
export M2M_DATA_DIR=... # Preprocessed/binarized data
```
and select `data_sample = 'generate'` in `m2m_interpretability.ipynb`.

You can select to evaluate interpretations using teacher forcing or free decoding by setting `teacher_forcing` variable in the notebook.
## Usage with bilingual model
Define the following environment variables before running `nmt_interpretability.ipynb` nmt notebook:

```bash
export EUROPARL_CKPT_DIR=...
export IWSLT14_DATA_DIR=...
export IWSLT14_CKPT_DIR=...
```