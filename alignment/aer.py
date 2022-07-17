from ctypes import alignment
from email.policy import default
import torch
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import itertools
from collections import Counter, defaultdict
from scipy.stats import spearmanr, pearsonr
import warnings
import string
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()
from wrappers.interactive import *
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils

from fairseq.data.multilingual.multilingual_utils import (
    EncoderLangtok,
    LangTokSpec,
    LangTokStyle,
    augment_dictionary,
    get_lang_tok,
)

import alignment.align as align

gold_dir = Path(os.environ['GOLD_ALIGNMENT_DATA_DIR'])

class aer():
    def __init__(self, args, mode_list):
        self.test_set_dir = args.test_set_dir
        self.src = args.src
        self.tgt = args.tgt
        self.tokenizer = args.tokenizer
        self.test_src_bpe = f'{args.test_set_dir}/test.{args.tokenizer}.{args.src}'
        self.test_tgt_bpe = f'{args.test_set_dir}/test.{args.tokenizer}.{args.tgt}'
        self.test_src_word = f'{args.test_set_dir}/test.{args.src}'
        self.test_tgt_word = f'{args.test_set_dir}/test.{args.tgt}'
        self.gold_alignment = gold_dir / "alignment.talp"
        self.model_name_save = args.model_name_save
        self.mode_list = mode_list
        self.num_layers = args.num_layers

    def extract_contribution_matrix(self, hub, model_name_save, contrib_type, pre_layer_norm=False):
        """Extract contribution matrix."""

        extract_matrix = []
                    
        ## TODO: avoid doing this
        #test_src_bpe = f'{test_set_dir}/test.{tokenizer}.{src}'
        with open(self.test_src_bpe, encoding="utf-8") as fbpe:
            # BPE source sentences
            src_bpe_sents = fbpe.readlines()

        for i in range(len(src_bpe_sents)):#len(src_bpe_sents)

            if i%200==0:
                print(i)

            sample = hub.get_interactive_sample(i, self.test_set_dir, self.src, self.tgt, self.tokenizer)
            src_tensor = sample['src_tensor']
            
            # if 'm2m' in model_name_save:
            #     src_tensor = sample['src_tensor']
            #     # src_lan_token = get_lang_tok(lang=hub.task.source_langs[0], lang_tok_style=LangTokStyle.multilingual.value)
            #     # idx_src_lan_token = hub.task.source_dictionary.index(src_lan_token)
            #     # src_tensor_forward = torch.cat([torch.tensor([idx_src_lan_token]).to(src_tensor.device),src_tensor])
            # else:
            #     src_tensor_forward = sample['src_tensor']
            tgt_tensor = sample['tgt_tensor']

            alti_model = hub.get_contribution_rollout(src_tensor, tgt_tensor, contrib_type,
                                                        norm_mode='min_sum',pre_layer_norm=pre_layer_norm)

            # Cross-attention
            cross_attn_contributions = alti_model['decoder.encoder_attn'].detach()                          
            cross_attn_contributions = cross_attn_contributions[:,:,:-1]#.detach() # leave out decoder residual

            # Cross-attention * encoder_alti
            encoder_alti = alti_model['encoder.self_attn'][-1].detach() # last layer encoder alti
            combined_alti_enc_cross = torch.matmul(cross_attn_contributions,encoder_alti.unsqueeze(0))

            # Total ALTI
            total_alti = alti_model['total'].detach()
            total_alti_pred_src = total_alti[:,:,:src_tensor.size(0)] # source sentence + </s>

            # Attention weights
            alti_model_attn_w = torch.squeeze(hub.get_contributions(src_tensor, tgt_tensor, contrib_type='attn_w',
                                                        norm_mode='sum_one',pre_layer_norm=pre_layer_norm)['decoder.encoder_attn']).detach()

            extract_matrix.append({"alti":total_alti_pred_src, "decoder.encoder_attn": cross_attn_contributions,
                                    "alti_enc_cross_attn": combined_alti_enc_cross, "attn_w":alti_model_attn_w})

        
        store_filename = f'./results/alignments/{model_name_save}/extracted_matrix.pkl'
        os.makedirs(os.path.dirname(store_filename), exist_ok=True)
        with open(store_filename, 'wb') as f:
                pickle.dump(extract_matrix, f)

    def extract_alignments(self, final_punc_mark=False):
        """Extract word-word alignments from contribution matrix (MAP)."""

        with open(self.test_src_bpe, encoding="utf-8") as fbpe:
            # BPE source sentences
            src_bpe_sents = fbpe.readlines()
        with open(self.test_tgt_bpe, encoding="utf-8") as fbpe:
            # BPE target sentences
            tgt_bpe_sents = fbpe.readlines()
        with open(self.test_src_word, encoding="utf-8") as fword:
            # Original source sentences
            src_word_sents = fword.readlines()
        with open(self.test_tgt_word, encoding="utf-8") as fword:
            # Original target sentences
            tgt_word_sents = fword.readlines()

        punc_string = string.punctuation
        punct_tok = ['▁'+punc for punc in punc_string]

        store_filename = f'./results/alignments/{self.model_name_save}/extracted_matrix.pkl'
        with open(store_filename, 'rb') as f:
                extract_matrix = pickle.load(f)

        for setting in ['AWO', 'AWI']:
            for mode in self.mode_list:
                print(mode)
                for l in range(self.num_layers):
                    with open(f'./results/alignments/{self.model_name_save}/hypothesis_{l}_{mode}_{setting}', 'w') as f:
                        for i in range(len(src_bpe_sents)):#len(src_bpe_sents)
                            src_bpe_sent = src_bpe_sents[i]
                            src_word_sent = src_word_sents[i]
                            splited_src_bpe_sent = src_bpe_sent.split()
                            splited_src_word_sent = src_word_sent.split()

                            tgt_bpe_sent = tgt_bpe_sents[i]
                            tgt_word_sent = tgt_word_sents[i]
                            splited_tgt_bpe_sent = tgt_bpe_sent.split()
                            splited_tgt_word_sent = tgt_word_sent.split()

                            if "m2m" in self.model_name_save:
                                splited_src_bpe_sent = ['__src__'] + splited_src_bpe_sent
                                splited_tgt_bpe_sent = ['__tgt__'] + splited_tgt_bpe_sent

                                splited_src_word_sent = ['__src__'] + splited_src_word_sent
                                splited_tgt_word_sent = ['__tgt__'] + splited_tgt_word_sent

                            src_len = len(src_word_sent.split())
                            
                            contrib_matrix = torch.squeeze(extract_matrix[i][mode][l]).detach().cpu().numpy()

                            if setting == "AWI":
                                contrib_matrix = contrib_matrix[list(range(1,len(contrib_matrix)))+[0]]

                            ## Word-word attention
                            source_sentence = splited_src_bpe_sent + ['</s>']
                            target_sentence = ['</s>'] + splited_tgt_bpe_sent
                            predicted_sentence = splited_tgt_bpe_sent + ['</s>']
                            
                            contrib_matrix, words_in, words_out = align.contrib_tok2words(
                            contrib_matrix,
                            tokens_in=(source_sentence + target_sentence),
                            tokens_out=predicted_sentence)

                            # Eliminate language tags
                            if "m2m" in self.model_name_save:
                                contrib_matrix = contrib_matrix[1:,1:]

                            # We don't consider alignment of EOS (target/column)
                            contrib_matrix = contrib_matrix[:-1]

                            # Assign final mark alignments (Chen et al., 2020)
                            if final_punc_mark:
                                if splited_tgt_bpe_sent[-1] in punct_tok:
                                    contrib_matrix[-1,:] = float('-inf')
                                    contrib_matrix[-1,-2] = float('+inf')
                            
                            #contrib_matrix[:,-1] = float('-inf')
                            
                            contrib_argmax = np.argmax(contrib_matrix, -1)

                            for t, s_a in enumerate(contrib_argmax):
                                if s_a != src_len:
                                    f.write("{}-{} ".format(t+1, s_a+1))
                            f.write("\n")


    def calculate_aer(self, setting="AWO"):
        """Calculate AER (alignment error rate), Precision and Recall for constructed alignments in the layer-level."""

        sure, possible = [], []

        with open(self.gold_alignment, 'r') as f:
            for line in f:
                sure.append(set())
                possible.append(set())

                for alignment_string in line.split():

                    sure_alignment = True if '-' in alignment_string else False
                    alignment_tuple = align.parse_single_alignment(alignment_string, reverse=True)

                    if sure_alignment:
                        sure[-1].add(alignment_tuple)
                    possible[-1].add(alignment_tuple)

        target_sentences = []
        with open(self.test_tgt_word, encoding="utf-8") as fe:
            for en in fe:
                target_sentences.append(en.split())

        source_sentences = []
        with open(self.test_src_word,encoding="utf-8") as fd:
            for de in fd:
                source_sentences.append(de.split())

        assert len(sure) == len(possible)
        assert len(target_sentences) == len(source_sentences)
        assert len(sure) == len(source_sentences)

        results = {}

        for mode in self.mode_list:
            metrics = defaultdict(list)
            for l in range(self.num_layers):
    
                hypothesis = []

                hypo_file = f'./results/alignments/{self.model_name_save}/hypothesis_{l}_{mode}_{setting}'

                with open(hypo_file) as f:
                    for line in f:
                        hypothesis.append(set())

                        for alignment_string in line.split():
                            alignment_tuple = align.parse_single_alignment(alignment_string)
                            hypothesis[-1].add(alignment_tuple)

                sum_a_intersect_p, sum_a_intersect_s, sum_s, sum_a = 4 * [0.0]

                for S, P, A in itertools.zip_longest(sure, possible, hypothesis):

                    sum_a += len(A)
                    sum_s += len(S)
                    sum_a_intersect_p += len(A.intersection(P))
                    sum_a_intersect_s += len(A.intersection(S))

                #precision = sum_a_intersect_p / sum_a
                recall = sum_a_intersect_s / sum_s
                aer = 1.0 - ((sum_a_intersect_p + sum_a_intersect_s) / (sum_a + sum_s))

                #metrics['precision'].append(precision)
                metrics['recall'].append(recall)
                metrics['aer'].append(aer)
                
            results[mode] = metrics
        

        return results