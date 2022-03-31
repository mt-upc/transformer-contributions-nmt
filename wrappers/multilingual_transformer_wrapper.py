import logging
import warnings
from functools import partial
from collections import defaultdict

# From hub.utils
import copy
from typing import Any, Dict, Iterator, List
from omegaconf import open_dict
from fairseq import utils
logger = logging.getLogger(__name__)

from fairseq.data.multilingual.multilingual_utils import (
    EncoderLangtok,
    LangTokSpec,
    LangTokStyle,
    augment_dictionary,
    get_lang_tok,
)


import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from wrappers.transformer_wrapper import FairseqTransformerHub
from fairseq.models.transformer import TransformerModel

from einops import rearrange

import matplotlib.pyplot as plt
import seaborn as sns


class FairseqMultilingualTransformerHub(FairseqTransformerHub):

    def __init__(self, cfg, task, models):
        super().__init__(cfg, task, models)
        self.eval()
        self.to("cuda" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def from_pretrained(cls, checkpoint_dir, checkpoint_file, data_name_or_path, source_lang, target_lang, lang_pairs):
        hub_interface = TransformerModel.from_pretrained(checkpoint_dir, checkpoint_file, data_name_or_path,
                                                        source_lang=source_lang,
                                                        target_lang=target_lang,
                                                        lang_pairs=lang_pairs,
                                                        fixed_dictionary =f'{checkpoint_dir}/model_dict.128k.txt'
                                                        )
        return cls(hub_interface.cfg, hub_interface.task, hub_interface.models)

    def decode(self, tensor, dictionary, as_string=False):
        tok = []
        for token in torch.squeeze(tensor):
                tok.append(dictionary[token])
        # tok = dictionary.string(tensor,'sentencepiece').split()
        if as_string:
            return ''.join(tok).replace('▁', ' ')
        else:
            return tok

    def get_sample(self, split, index):

        if split not in self.task.datasets.keys():
            self.task.load_dataset(split)

        src_tensor = self.task.dataset(split)[index]['source']
        src_tok = self.decode(src_tensor, self.task.source_dictionary)
        src_sent = self.decode(src_tensor, self.task.source_dictionary, as_string=True)

        # Adding EOS token to beggining of target sentence
        tgt_tensor = self.task.dataset(split)[index]['target']
        #dataset=task.datasets[split + ':' + langdir],
        tgt_tensor = torch.cat([torch.tensor([self.task.target_dictionary.eos_index]),
                                tgt_tensor[:-1]
                                ]).to(tgt_tensor.device)
        tgt_tok = self.decode(tgt_tensor, self.task.target_dictionary)
        tgt_sent = self.decode(tgt_tensor, self.task.target_dictionary, as_string=True)

        return src_sent, src_tok, src_tensor, tgt_sent, tgt_tok, tgt_tensor

    def get_interactive_sample(self, i, test_set_dir, src, tgt, tokenizer):
        """Get interactive sample from tokenized and original word files."""

        test_src_bpe = f'{test_set_dir}/test.{tokenizer}.{src}'
        test_tgt_bpe = f'{test_set_dir}/test.{tokenizer}.{tgt}'
        test_src_word = f'{test_set_dir}/test.{src}'
        test_tgt_word = f'{test_set_dir}/test.{tgt}'

        with open(test_src_bpe, encoding="utf-8") as fbpe:
            # BPE source sentences
            src_bpe_sents = fbpe.readlines()
        with open(test_tgt_bpe, encoding="utf-8") as fbpe:
            # BPE target sentences
            tgt_bpe_sents = fbpe.readlines()
        with open(test_src_word, encoding="utf-8") as fword:
            # Original source sentences
            src_word_sents = fword.readlines()
        with open(test_tgt_word, encoding="utf-8") as fword:
            # Original target sentences
            tgt_word_sents = fword.readlines()

        src_word_sent = src_word_sents[i]
        tgt_word_sent = tgt_word_sents[i]

        src_tok_str = src_bpe_sents[i].strip() # removes leading and trailing whitespaces
        src_tok = src_tok_str.split()

        tgt_tok_str = tgt_bpe_sents[i].strip() # removes leading and trailing whitespaces
        tgt_tok = tgt_tok_str.split()

        # M2M generate has --decoder-langtok --encoder-langtok src
        # adds source language token to the source sentence
        # adds target language token to the target sentence
        src_lan_token = get_lang_tok(lang=self.task.source_langs[0], lang_tok_style=LangTokStyle.multilingual.value)
        tgt_lan_token = get_lang_tok(lang=self.task.target_langs[0], lang_tok_style=LangTokStyle.multilingual.value)
        idx_src_lan_token = self.task.source_dictionary.index(src_lan_token)
        idx_tgt_lan_token = self.task.target_dictionary.index(tgt_lan_token)

        src_tok = [src_lan_token] + src_tok + [self.task.source_dictionary[self.task.source_dictionary.eos_index]]
        tgt_tok = [self.task.target_dictionary[self.task.target_dictionary.eos_index]] + [tgt_lan_token] + tgt_tok

        src_tensor = torch.tensor([self.task.source_dictionary.index(s) for s in src_tok])
        tgt_tensor = torch.tensor([self.task.target_dictionary.index(t) for t in tgt_tok])

        if test_src_word and test_tgt_word:
            src_word_sent = src_word_sents[i]
            tgt_word_sent = tgt_word_sents[i]
            return src_word_sent, src_tok, src_tok_str, src_tensor, tgt_word_sent, tgt_tok, tgt_tok_str, tgt_tensor

        return None, src_tok, src_tok_str, src_tensor, None, tgt_tok, tgt_tok_str, tgt_tensor

    def trace_forward(self, src_tensor, tgt_tensor):
        r"""Forward-pass through the model.
        Args:
            src_tensor (`tensor`):
                Source sentence tensor.
            tgt_tensor (`tensor`):
                Target sentence tensor (teacher forcing).
        Returns:
            model_output ('tuple'):
                output of the model.
            log_probs:
                log probabilities output by the model.
            encoder_output ('dict'):
                dictionary with 'encoder_out', 'encoder_padding_mask', 'encoder_embedding',
                                'encoder_states', 'src_tokens', 'src_lengths', 'attn_weights'.
            layer_inputs:
                dictionary with the input of the modeules of the model.
            layer_outputs:
                dictionary with the input of the modeules of the model.
        """
        #self.zero_grad()
        with torch.no_grad():

            layer_inputs = defaultdict(list)
            layer_outputs = defaultdict(list)

            def save_activation(name, mod, inp, out):
                layer_inputs[name].append(inp)
                layer_outputs[name].append(out)

            handles = {}

            for name, layer in self.named_modules():
                handles[name] = layer.register_forward_hook(partial(save_activation, name))
            
            src_tensor = src_tensor.unsqueeze(0).to(self.device)

            # tgt_tensor = torch.cat([
            #     torch.tensor([self.task.target_dictionary.eos_index]),
            #     tgt_tensor[:-1]
            # ]).unsqueeze(0).to(self.device)
            tgt_tensor = tgt_tensor.unsqueeze(0).to(self.device)

            model_output, encoder_out = self.models[0](src_tensor, src_tensor.size(-1), tgt_tensor, )

            log_probs = self.models[0].get_normalized_probs(model_output, log_probs=True, sample=None)
            
            for k, v in handles.items():
                handles[k].remove()
            
            return model_output, log_probs, encoder_out, layer_inputs, layer_outputs

    def generate(
        self,
        tokenized_sentences: List[torch.LongTensor],
        beam: int = 5,
        verbose: bool = False,
        skip_invalid_size_inputs=False,
        inference_step_args=None,
        prefix_allowed_tokens_fn=None,
        **kwargs
        ) -> List[List[Dict[str, torch.Tensor]]]:
        if torch.is_tensor(tokenized_sentences) and tokenized_sentences.dim() == 1:
            return self.generate(
                tokenized_sentences.unsqueeze(0), beam=beam, verbose=verbose, **kwargs
            )[0]

        # build generator using current args as well as any kwargs
        gen_args = copy.deepcopy(self.cfg.generation)
        with open_dict(gen_args):
            gen_args.beam = beam
            for k, v in kwargs.items():
                setattr(gen_args, k, v)
        generator = self.task.build_generator(
            self.models,
            gen_args,
            #prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )
        inference_step_args = inference_step_args or {}
        results = []
        for batch in self._build_batches(tokenized_sentences, skip_invalid_size_inputs):
            batch = utils.apply_to_sample(lambda t: t.to(self.device), batch)
            translations = self.task.inference_step(
                generator, self.models, batch, **inference_step_args
            )
            for id, hypos in zip(batch["id"].tolist(), translations):
                results.append((id, hypos))

        # sort output to match input order
        outputs = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]

        if verbose:

            def getarg(name, default):
                return getattr(gen_args, name, getattr(self.cfg, name, default))

            for source_tokens, target_hypotheses in zip(tokenized_sentences, outputs):
                src_str_with_unk = self.string(source_tokens)
                logger.info("S\t{}".format(src_str_with_unk))
                for hypo in target_hypotheses:
                    hypo_str = self.decode(hypo["tokens"],self.task.target_dictionary)
                    logger.info("H\t{}\t{}".format(hypo["score"], hypo_str))
                    logger.info(
                        "P\t{}".format(
                            " ".join(
                                map(
                                    lambda x: "{:.4f}".format(x),
                                    hypo["positional_scores"].tolist(),
                                )
                            )
                        )
                    )
                    if hypo["alignment"] is not None and getarg(
                        "print_alignment", False
                    ):
                        logger.info(
                            "A\t{}".format(
                                " ".join(
                                    [
                                        "{}-{}".format(src_idx, tgt_idx)
                                        for src_idx, tgt_idx in hypo["alignment"]
                                    ]
                                )
                            )
                        )
        return outputs