import torch
from collections import defaultdict
from fairseq.data.multilingual.multilingual_utils import (
    EncoderLangtok,
    LangTokSpec,
    LangTokStyle,
    augment_dictionary,
    get_lang_tok,
)

def get_greedy_decoding(hub, src_tensor, beam, inference_step_args=None):
    tgt_tensor_free = []
    # Compute beam search (beam = 1 greedy decoding)
    for pred in hub.generate(src_tensor, beam, inference_step_args = inference_step_args):
        tgt_tensor_free.append(pred['tokens'])
        translation = hub.decode(pred['tokens'][:-1], hub.task.target_dictionary, as_string=True)

    hypo = 0 # first hypothesis
    tgt_tensor = tgt_tensor_free[hypo]
    
    # We add eos token at the beginning of sentence and delete it from the end
    tgt_tensor = torch.cat([torch.tensor([hub.task.target_dictionary.eos_index]).to(tgt_tensor.device),
                    tgt_tensor[:-1]
                ]).to(tgt_tensor.device)
    #print('tgt_tensor inside', tgt_tensor.size())
    return tgt_tensor, translation

def contribution_source(total_rollout, src_tensor):
    # Get total source contribution
    src_total_alti = total_rollout[-1][:,:src_tensor.size(0)].sum(dim=-1)
    # We delete the first prediction (only source and bos contribution)
    src_total_alti = src_total_alti[1:]
    return src_total_alti
    

def get_translation(hub, i, args, perturb_type='src', prefix_subwords=None, contributions=True):
    """Get translation and total source contribution by ALTI"""
    
    if prefix_subwords is not None:
        # Add token to beginning of target sentence (prefix)
        if perturb_type == 'tgt':
            # Convert prefix word to token indices, and tensor (required for generate)
            token_idx = hub.task.target_dictionary.index(prefix_subwords)
            prefix_tokens = torch.tensor([[token_idx]]).to('cuda')

            inference_step_args={'prefix_tokens': prefix_tokens}

            # Get sample from test set
            sample = hub.get_interactive_sample(i, args.test_set_dir, args.src, args.tgt, args.tokenizer)
            src_tensor = sample['src_tensor']

        else:
            # Get sample from test set with prefix token
            # Add token to beginning of source sentence
            sample = hub.get_interactive_sample(i, args.test_set_dir, args.src, args.tgt, args.tokenizer, hallucination=prefix_subwords)
            src_tensor = sample['src_tensor']
            #tgt_tensor = sample['tgt_tensor']
            inference_step_args = None

        tgt_tensor, translation = get_greedy_decoding(hub, src_tensor, args.beam, inference_step_args)

    else:
        # No hallucination
        sample = hub.get_interactive_sample(i, args.test_set_dir, args.src, args.tgt, args.tokenizer)
        src_tensor = sample['src_tensor']
        tgt_tensor, translation = get_greedy_decoding(hub, src_tensor, args.beam, inference_step_args=None)

    # src_lan_token = get_lang_tok(lang=hub.task.source_langs[0], lang_tok_style=LangTokStyle.multilingual.value)
    # idx_src_lan_token = hub.task.source_dictionary.index(src_lan_token)
    # src_tensor_forward = torch.cat([torch.tensor([idx_src_lan_token]).to(src_tensor.device),src_tensor])
    
    # Compute contributions with ALTI if required
    if contributions:
        total_rollout = hub.get_contribution_rollout(src_tensor, tgt_tensor, 'l1',
                                                    norm_mode='min_sum',
                                                    pre_layer_norm=args.pre_layer_norm)['total'].detach()
        src_total_alti = contribution_source(total_rollout, src_tensor)
    else:
        src_total_alti = None

    return translation, src_total_alti

def lang_toks_contrib(total_rollout_layer, src_tensor):
    return total_rollout_layer[:,0], total_rollout_layer[:,src_tensor.size(0)+1] # first column

def model_analysis(hub, sentences, args, eos_res_corr=False, lang_toks=False, src_contrib=False):
    eos_list = []
    res_list = []
    punct_list = []
    src_alti_list = []

    eos_dict = defaultdict(list)
    res_dict = defaultdict(list)
    res_sum_dict = defaultdict(list)

    lang_tok_src = []
    lang_tok_tgt = []
    i = 0
    counter = 0
    number_hallucinations = 0
    for i in sentences:
        counter += 1
        if counter%100==0:
            print(counter)
        sample = hub.get_interactive_sample(i, args.test_set_dir, args.src, args.tgt, args.tokenizer)
        src_tensor = sample['src_tensor']
        src_tok = sample['src_tok']
        src_len = len(src_tok)
        tgt_tensor, _ = get_greedy_decoding(hub, src_tensor, args.beam)
        src_lan_token = get_lang_tok(lang=hub.task.source_langs[0], lang_tok_style=LangTokStyle.multilingual.value)
        idx_src_lan_token = hub.task.source_dictionary.index(src_lan_token)
        src_tensor_forward = torch.cat([torch.tensor([idx_src_lan_token]).to(src_tensor.device),src_tensor])

        if(tgt_tensor.size(0)>1.8*src_len):
            print('Hallucination')
            number_hallucinations += 1
            tgt_tensor = tgt_tensor[:int(1.5*src_len)]
            

        # Analyze EOS column and Residual contribution
        if eos_res_corr:
            # Attn weights cross-attention
            cross_attn_weights = torch.squeeze(hub.get_contributions(src_tensor_forward, tgt_tensor, 'attn_w',
                                                                    norm_mode='sum_one',pre_layer_norm=args.pre_layer_norm)['decoder.encoder_attn'])
            cross_attn_weights = cross_attn_weights.detach().cpu()

            cross_attn_contributions = torch.squeeze(hub.get_contributions(src_tensor_forward, tgt_tensor,
                                                                            'l1', norm_mode='min_sum',
                                                                            pre_layer_norm=args.pre_layer_norm)['decoder.encoder_attn'])
            cross_attn_contributions = cross_attn_contributions.detach().cpu()

            for layer in range(args.num_layers):
                eos_column = cross_attn_weights[layer][:,-1].tolist()
                eos_dict[layer].append(eos_column)
                residual_column = cross_attn_contributions[layer][:,-1].tolist()
                res_dict[layer].append(residual_column)

        # ## Analyze Lang toks
        # if lang_toks:
        #     total_rollout = hub.get_contribution_rollout(src_tensor_forward, tgt_tensor, 'l1',
        #                                         norm_mode='min_sum', pre_layer_norm=args.pre_layer_norm)['total'].detach()
        #     total_rollout_layer = total_rollout[-1] # last layer alti+
        #     lang_tok_src_col, lang_tok_tgt_col = lang_toks_contrib(total_rollout_layer, src_tensor_forward)
        #     lang_tok_src.append(lang_tok_src_col[1:].tolist())
        #     lang_tok_tgt.append(lang_tok_tgt_col[2:].tolist())

        if src_contrib:
            total_rollout = hub.get_contribution_rollout(src_tensor_forward, tgt_tensor, 'l1',
                                                        norm_mode='min_sum',
                                                        pre_layer_norm=args.pre_layer_norm)['total'].detach()

            src_total_alti = contribution_source(total_rollout, src_tensor_forward)
            src_alti_list.append(torch.mean(src_total_alti).item())# src_total_alti.tolist()

    if src_contrib:
        return src_alti_list, number_hallucinations
    elif eos_res_corr:
        return eos_dict, res_dict