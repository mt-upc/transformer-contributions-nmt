# ALTI+

This repository includes the code from the paper [Ferrando et al., 2022](https://arxiv.org/abs/2205.11631).

We extend ALTI method presented in [Ferrando et al., 2022](https://arxiv.org/abs/2203.04212) to the encoder-decoder setting.
## Usage with M2M model

Follow fairseq [M2M](https://github.com/pytorch/fairseq/tree/main/examples/m2m_100) instructions and download the models (412M and 1.2B), dictionary, and SPM tokenizer in `M2M_CKPT_DIR`. Then, set the environmental variable:

```bash
export M2M_CKPT_DIR=...
````

`m2m_interpretability.ipynb` can be run in generate or interactive mode. Interactive mode reads from already tokenized files.

To use it in interactive mode, tokenize your own test data and select `data_sample = 'interactive'` in `m2m_interpretability.ipynb`.

Already tokenized data from [FLORES-101](https://github.com/facebookresearch/flores/blob/main/flores200/README.md) and [De-En Gold Alignment](https://www-i6.informatik.rwth-aachen.de/goldAlignment/) for testing the notebook is available in `./data`.

Using FLORES-101 as an example, create the environmental variable where to store the dataset:

```bash
export M2M_DATA_DIR=...
```

Download the dataset:
```bash
cd $M2M_DATA_DIR
wget https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz
tar -xf flores101_dataset.tar.gz
```
Then, specify the `TRG_LANG_CODE`  language and run:
```bash
TRG_LANG_CODE=spa

python path_to_fairseq/fairseq/scripts/spm_encode.py \
    --model $M2M_CKPT_DIR/spm.128k.model \
    --output_format=piece \
    --inputs=$M2M_DATA_DIR/flores101_dataset/devtest/${TRG_LANG_CODE}.devtest \
    --outputs=./data/flores/test.spm.${TRG_LANG_CODE}

cp $M2M_DATA_DIR/flores101_dataset/devtest/${TRG_LANG_CODE}.devtest ./data/flores/test.${TRG_LANG_CODE}
```

You can select to evaluate interpretations using teacher forcing or free decoding/beam search by setting `teacher_forcing` variable in the notebook.

For your particular multilingual NMT model you may be interested in modifying `prepare_input_decoder` and `prepare_input_encoder` functions in `./wrappers/multilingual_transformer_wrapper.py` to handle properly the language tags in the encoder and decoder inputs. Currently it follows M2M model structure. This corresponds to `--decoder-langtok` and `--encoder-langtok src` parameters in `fairseq-generate`.

To use the notebook in generate mode, provide the path to the binarized data after running `fairseq-preprocess` and select `data_sample = 'generate'` in `m2m_interpretability.ipynb`.

*Since the method needs to access the activations of keys, queries, and values from the attention mechanism, we need to make fairseq avoid using PyTorch's attention implementation (F.multi_head_attention_forward) by commenting this part of the code in `fairseq/fairseq/modules/multihead_attention.py`:

<p align="center"><br>
<img src="./img/comment.png" class="center" title="paper logo" width="300"/>
</p><br>