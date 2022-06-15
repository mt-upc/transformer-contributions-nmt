# ALTI+

This repository includes the code from the paper [Ferrando et al., 2022](https://arxiv.org/abs/2205.11631).

We extend ALTI method presented in [Ferrando et al., 2022](https://arxiv.org/abs/2203.04212) to the encoder-decoder setting.
## Usage with M2M model

Follow fairseq [M2M](https://github.com/pytorch/fairseq/tree/main/examples/m2m_100) instructions and download the models (412M and 1.2B), dictionary, and SPM tokenizer in `M2M_CKPT_DIR`. Then, set the environmental variable:

```bash
export M2M_CKPT_DIR=... # Path to checkpoint, dictionary, and SPM
````

`m2m_interpretability.ipynb` can be run in generate or interactive mode. Generate mode gets samples from the binarized files in the folder provided to `fariseq-preprcess`. Interactive mode reads from already tokenized files.

### To use it in interactive mode, tokenize your own test data using:
```bash
python /path/to/fairseq/scripts/spm_encode.py \
    --model spm.128k.model \
    --output_format=piece \
    --inputs=/path/to/input/file/here \
    --outputs=/path/to/output/file/here
```

and select `data_sample = 'interactive'` in `m2m_interpretability.ipynb`.

Already tokenized data from [FLORES-101](https://github.com/facebookresearch/flores) and  [De-En Gold Alignment](https://www-i6.informatik.rwth-aachen.de/goldAlignment/) for testing the notebook is available in `./data`.

You can select to evaluate interpretations using teacher forcing or free decoding by setting `teacher_forcing` variable in the notebook.

You may be interested in modifying `prepare_input_decoder` and `prepare_input_encoder` functions to handle properly the language tags in the encoder and decoder inputs. Currently follows M2M model structure. This corresponds to `--decoder-langtok` and `--encoder-langtok src` parameters in `fairseq-generate`.

To use the notebook in generate mode, provide the path to the binarized data by setting the environment variable `M2M_DATA_DIR`:
```bash
export M2M_DATA_DIR=... # Preprocessed/binarized data
```
and select `data_sample = 'generate'` in `m2m_interpretability.ipynb`.
