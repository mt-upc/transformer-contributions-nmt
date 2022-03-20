import logging
import warnings
from functools import partial
from collections import defaultdict

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.hub_utils import GeneratorHubInterface
from fairseq.models.transformer import TransformerModel

from einops import rearrange

import matplotlib.pyplot as plt
import seaborn as sns

rc={'font.size': 12, 'axes.labelsize': 10, 'legend.fontsize': 10.0,
    'axes.titlesize': 24, 'xtick.labelsize': 24, 'ytick.labelsize': 24,
    'axes.linewidth': .5, 'figure.figsize': (12,12)}
plt.rcParams.update(**rc)


class FairseqTransformerHub(GeneratorHubInterface):
    ATTN_MODULES = ['encoder.self_attn',
                    'decoder.self_attn',
                    'decoder.encoder_attn']

    def __init__(self, cfg, task, models):
        super().__init__(cfg, task, models)
        self.eval()
        self.to("cuda" if torch.cuda.is_available() else "cpu")
    
    @classmethod
    def from_pretrained(cls, checkpoint_dir, checkpoint_file, data_name_or_path):
        hub_interface = TransformerModel.from_pretrained(checkpoint_dir, checkpoint_file, data_name_or_path)
        return cls(hub_interface.cfg, hub_interface.task, hub_interface.models)

    def encode(self, sentence, dictionary):
        raise NotImplementedError()
    
    def decode(self, tensor, dictionary, as_string=False):
        tok = dictionary.string(tensor).split()
        if as_string:
            return ''.join(tok).replace('▁', ' ')
        else:
            return tok
    
    def get_sample(self, split, index):

        if split not in self.task.datasets.keys():
            self.task.load_dataset(split)

        src_tensor = self.task.dataset(split)[index]['source']
        src_tok = self.decode(src_tensor, self.task.src_dict)
        src_sent = self.decode(src_tensor, self.task.src_dict, as_string=True)

        tgt_tensor = self.task.dataset(split)[index]['target']
        tgt_tok = self.decode(tgt_tensor, self.task.tgt_dict)
        tgt_sent = self.decode(tgt_tensor, self.task.tgt_dict, as_string=True)
        
    
        return src_sent, src_tok, src_tensor, tgt_sent, tgt_tok, tgt_tensor

    def get_lm_sample(self, split, index):
        if split not in self.task.datasets.keys():
            self.task.load_dataset(split)

        src_tensor = self.task.dataset(split)[index]['source']
        src_tok = self.decode(src_tensor, self.task.source_dictionary)
        src_sent = self.decode(src_tensor, self.task.source_dictionary, as_string=True)

        return src_sent, src_tok, src_tensor


    
    def parse_module_name(self, module_name):
        """ Returns (enc_dec, layer, module)"""
        parsed_module_name = module_name.split('.')
        if not isinstance(parsed_module_name, list):
            parsed_module_name = [parsed_module_name]
            
        if len(parsed_module_name) < 1 or len(parsed_module_name) > 3:
            raise AttributeError(f"'{module_name}' unknown")
            
        if len(parsed_module_name) > 1:
            try:
                parsed_module_name[1] = int(parsed_module_name[1])
            except ValueError:
                parsed_module_name.insert(1, None)
            if len(parsed_module_name) < 3:
                parsed_module_name.append(None)
        else:
            parsed_module_name.extend([None, None])

        return parsed_module_name
    
    def get_module(self, module_name):
        e_d, l, m = self.parse_module_name(module_name)
        module = getattr(self.models[0], e_d)
        if l is not None:
            module = module.layers[l]
            if m is not None:
                module = getattr(module, m)
        else:
            if m is not None:
                raise AttributeError(f"Cannot get'{module_name}'")

        return module

    def trace_forward(self, src_tensor, tgt_tensor):
        self.zero_grad()

        layer_inputs = defaultdict(list)
        layer_outputs = defaultdict(list)

        def save_activation(name, mod, inp, out):
            layer_inputs[name].append(inp)
            layer_outputs[name].append(out)

        handles = {}

        for name, layer in self.named_modules():
            handles[name] = layer.register_forward_hook(partial(save_activation, name))
        
        src_tensor = src_tensor.unsqueeze(0).to(self.device)
        tgt_tensor = torch.cat([
            torch.tensor([self.task.tgt_dict.eos_index]),
            tgt_tensor[:-1]
        ]).unsqueeze(0).to(self.device)

        model_output, encoder_out = self.models[0](src_tensor, src_tensor.size(-1), tgt_tensor, )

        log_probs = self.models[0].get_normalized_probs(model_output, log_probs=True, sample=None)
        
        for k, v in handles.items():
            handles[k].remove()
        
        return model_output, log_probs, encoder_out, layer_inputs, layer_outputs

    def trace_lm_forward(self, tgt_tensor):
        self.zero_grad()

        layer_inputs = defaultdict(list)
        layer_outputs = defaultdict(list)

        def save_activation(name, mod, inp, out):
            layer_inputs[name].append(inp)
            layer_outputs[name].append(out)

        handles = {}

        for name, layer in self.named_modules():
            handles[name] = layer.register_forward_hook(partial(save_activation, name))
        
        tgt_tensor = tgt_tensor.unsqueeze(0).to(self.device)

        model_output, encoder_out = self.models[0](tgt_tensor)

        log_probs = self.models[0].get_normalized_probs(model_output, log_probs=True, sample=None)
        
        for k, v in handles.items():
            handles[k].remove()
        
        return model_output, log_probs, encoder_out, layer_inputs, layer_outputs

    def __get_attn_weights_module(self, layer_outputs, module_name):
        enc_dec_, l, attn_module_ = self.parse_module_name(module_name)
        
        attn_module = self.get_module(module_name)
        num_heads = attn_module.num_heads
        head_dim = attn_module.head_dim

        k = layer_outputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}.k_proj"][0]
        q = layer_outputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}.q_proj"][0]

        q, k = map(
            lambda x: rearrange(
                x,
                't b (n_h h_d) -> (b n_h) t h_d',
                n_h=num_heads,
                h_d=head_dim
            ),
            (q, k)
        )

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if enc_dec_ == 'decoder' and attn_module_ == 'self_attn':
            tri_mask = torch.triu(torch.ones_like(attn_weights), 1).bool()
            attn_weights[tri_mask] = -1e9

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = rearrange(
            attn_weights,
            '(b n_h) t_q t_k -> b n_h t_q t_k',
            n_h=num_heads
        )
        return attn_weights

    def normalize_contrib(self, x, mode=None, temperature=0.5, resultant_norm=None):
        if mode == 'min_max':
            # Min-max normalization (higher layers don't affect)
            x_min = x.min(-1, keepdim=True)[0]
            x_max = x.max(-1, keepdim=True)[0]
            x_norm = (x - x_min) / (x_max - x_min)
            x_norm = x_norm / x_norm.sum(dim=-1, keepdim=True)
        elif mode == 'softmax':
            x_norm = F.softmax(x / temperature, dim=-1)
        elif mode == 'sum_one':
            x_norm = x / x.sum(dim=-1, keepdim=True)
            # x_norm = x_norm.clamp(min=0)
        elif mode == 'min_sum':
            if resultant_norm == None:
                x_min = x.min(-1, keepdim=True)[0]
                # x_max = x.max(-1, keepdim=True)
                x_norm = x + torch.abs(x_min)
                x_norm = x_norm / x_norm.sum(dim=-1, keepdim=True)
            else:
                x_norm = x + torch.abs(resultant_norm.unsqueeze(1))
                x_norm = torch.clip(x_norm,min=0)
                #x_norm = torch.abs(x_norm)
                x_norm = x_norm / x_norm.sum(dim=-1,keepdim=True)
        elif mode == 'max_substract':
            x_min = x.min(1, keepdim=True)
            # x_max = x.max(-1, keepdim=True)
            x_norm = x - torch.abs(x_min)
            x_norm = x_norm / x_norm.sum(dim=-1, keepdim=True)
        elif mode is None:
            x_norm = x
        else:
            raise AttributeError(f"Unknown normalization mode '{mode}'")
        return x_norm
    
    def __get_contributions_module(self, layer_inputs, layer_outputs, contrib_type, contrib_inc_biases, result_inc_biases, module_name):
        enc_dec_, l, attn_module_ = self.parse_module_name(module_name)
        attn_w = self.__get_attn_weights_module(layer_outputs, module_name)
        
        def l_transform(x, w_ln):
            ln_param_transf = torch.diag(w_ln)
            ln_mean_transf = torch.eye(w_ln.size(0)).to(w_ln.device) - \
                1 / w_ln.size(0) * torch.ones_like(ln_param_transf).to(w_ln.device)

            out = torch.einsum(
                '... e , e f , f g -> ... g',
                x,
                ln_mean_transf,
                ln_param_transf
            )
            return out
        attn_module = self.get_module(module_name)
        w_o = attn_module.out_proj.weight
        b_o = attn_module.out_proj.bias
        
        ln = self.get_module(f'{module_name}_layer_norm')
        w_ln = ln.weight
        b_ln = ln.bias
        eps_ln = ln.eps
        
        in_q = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}.q_proj"][0][0].transpose(0, 1)
        in_v = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}.v_proj"][0][0].transpose(0, 1)
        t_q = in_q.size(1)
        t_v = in_v.size(1)

        if "self_attn" in attn_module_:
            residual_ = torch.diag_embed(in_q.transpose(-1, -2), dim1=-3, dim2=-2)
            b_o_ = (b_o * torch.eye(t_q).unsqueeze(-1).to(b_o.device))
            b_ln_ = (b_ln * torch.eye(t_q).unsqueeze(-1).to(b_ln.device))
        else:
            # print('in_q',in_q.size())
            # print('in_v',in_v.size())
            #b_o_ = (b_o * torch.eye(t_q).unsqueeze(-1).to(b_o.device))
            #b_ln_ = (b_ln * torch.eye(t_q).unsqueeze(-1).to(b_ln.device))
            #residual_ = in_q.unsqueeze(-2) / t_v
            #b_o_ = b_o / t_v
            #b_ln_ = b_ln / t_v
            ## Javi
            residual_ = in_q

        v = attn_module.v_proj(in_v)
        v = rearrange(
            v,
            'b t_v (n_h h_d) -> b n_h t_v h_d',
            n_h=attn_module.num_heads,
            h_d=attn_module.head_dim
        )

        w_o = rearrange(
            w_o,
            'out_d (n_h h_d) -> n_h h_d out_d',
            n_h=attn_module.num_heads,
        )

        attn_v_wo = torch.einsum(
            'b h q k , b h k e , h e f -> b q k f',
            attn_w,
            v,
            w_o
        )
        if "self_attn" in attn_module_:
            out_qv_pre_ln = attn_v_wo + residual_# + b_o_
        else:
            attn_v_wo = torch.cat((attn_v_wo,residual_.unsqueeze(-2)),dim=2)
            out_qv_pre_ln = attn_v_wo
        out_q_pre_ln = out_qv_pre_ln.sum(-2) + b_o

        out_q_pre_ln_th = layer_inputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}_layer_norm"][0][0].transpose(0, 1)
        assert torch.dist(out_q_pre_ln_th, out_q_pre_ln).item() < 1e-3 * out_q_pre_ln.numel()

        if "self_attn" in attn_module_:
            ln_std_coef = 1/(out_q_pre_ln + eps_ln).std(-1).view(-1, 1, 1)
            out_qv = ln_std_coef * l_transform(attn_v_wo, w_ln) + ln_std_coef * l_transform(residual_, w_ln)  + ln_std_coef * l_transform(b_o_, w_ln) +  b_ln_
            out_q = out_qv.sum(-2)
            #print('out_q',out_q[0,0,:10])
        else:
            ln_std_coef = 1/(out_q_pre_ln + eps_ln).std(-1) # (1,9)
            ln_std_coef = ln_std_coef.view(1,-1, 1) # (1,9,1,1)
            out_qv = ln_std_coef.unsqueeze(-1) * l_transform(attn_v_wo, w_ln) # l_transform(attn_v_wo, w_ln) -> [1, 9, 14, 512]
            out_q = out_qv.sum(-2) + (ln_std_coef.unsqueeze(-1) * l_transform(b_o, w_ln)).squeeze(2) # [1, 9, 512]
            

        

        #out_q_th_1 = ln_std_coef.squeeze(-1) * l_transform(out_q_pre_ln, w_ln) + b_ln
        out_q_th_2 = layer_outputs[f"models.0.{enc_dec_}.layers.{l}.{attn_module_}_layer_norm"][0].transpose(0, 1)
        #print('real_output',out_q_th_2[0,0,:10])
        
        # assert torch.dist(out_q_th_2, out_q).item() < 1e-3 * out_q.numel()
        # assert torch.dist(out_q_th_2, out_q).item() < 1e-3 * out_q.numel()

        if contrib_inc_biases:
            contributors = out_qv
        else:
            if "self_attn" in attn_module_:
                contributors = out_qv - ln_std_coef * l_transform(b_o_, w_ln) - b_ln_
            else:
                contributors = out_qv - ln_std_coef.unsqueeze(-1) * l_transform(b_o, w_ln) # [1, 9, 14, 512]

        if result_inc_biases:
            if "self_attn" in attn_module_:
                resultant = out_q.unsqueeze(-2)
            else:
                resultant = (out_q + b_ln).unsqueeze(2) # [1, 9, 1, 512]
                #print('resultant',resultant[0,0,0,:10])
        else:
            resultant = out_q.unsqueeze(-2) - \
                (ln_std_coef * l_transform(b_o_, w_ln)).sum(-2, keepdim=True) - b_ln

        if contrib_type == 'l1':
            contributions = -F.pairwise_distance(contributors, resultant, p=1)
            resultants_norm = torch.norm(torch.squeeze(resultant),p=1,dim=-1)
        elif contrib_type == 'l2':
            contributions = -F.pairwise_distance(contributors, resultant, p=2)
        else:
            raise ArgumentError(f"contribution_type '{contrib_type}' unknown")
        
        #if enc_dec_ == 'decoder' and attn_module_ == 'self_attn':
        #    tri_contr = contributions.tril(0)
        #    max_contr = contributions.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
        #    min_tri_contr = (tri_contr + torch.ones_like(contributions)\
        #        .triu(1).mul(max_contr)).min(-1, keepdim=True)[0].min(-2, keepdim=True)[0]
        #    contributions = tri_contr + torch.ones_like(contributions).triu(1).mul(min_tri_contr)

        return contributions, resultants_norm
    
    def get_contributions(self, src_tensor, tgt_tensor, contrib_type='l1', norm_mode=None, contrib_inc_biases=False, result_inc_biases=True):
        contributions_all = defaultdict(list)
        _, _, _, layer_inputs, layer_outputs = self.trace_forward(src_tensor, tgt_tensor)
        
        if contrib_type == 'attn_w':
            f = partial(self.__get_attn_weights_module, layer_outputs)
        else:
            f = partial(
                self.__get_contributions_module,
                layer_inputs,
                layer_outputs,
                contrib_type,
                contrib_inc_biases,
                result_inc_biases
            )

        
        for attn in self.ATTN_MODULES:
            enc_dec_, _, attn_module_ = self.parse_module_name(attn)
            enc_dec = self.get_module(enc_dec_)

            for l in range(len(enc_dec.layers)):
                contributions, resultant_norms = f(attn.replace('.', f'.{l}.'))
                contributions = self.normalize_contrib(contributions, norm_mode, resultant_norm=resultant_norms).unsqueeze(1)
                # Mask upper triangle of decoder self-attention matrix (and normalize)
                # if attn == 'decoder.self_attn':
                #     contributions = torch.tril(torch.squeeze(contributions,dim=1))
                #     contributions = contributions / contributions.sum(dim=-1, keepdim=True)
                #     contributions = contributions.unsqueeze(1)
                contributions_all[attn].append(contributions)
        contributions_all = {k: torch.cat(v, dim=1) for k, v in contributions_all.items()}
        return contributions_all

    def get_contribution_rollout(self, src_tensor, tgt_tensor, contrib_type='l1', norm_mode='min_sum', **contrib_kwargs):
        c = self.get_contributions(src_tensor, tgt_tensor, contrib_type, norm_mode, **contrib_kwargs)
        if contrib_type == 'attn_w':
            c = {k: v.sum(2) for k, v in c.items()}
        c_roll = defaultdict(list)
        enc_sa = 'encoder.self_attn'
        _, layers, _, t_in = c[enc_sa].size()

        c_enc_sa_rollout = torch.eye(t_in).view(1, t_in, t_in).to(c[enc_sa].device)
        for l in range(layers):
            c_enc_sa_rollout = torch.einsum(
                'b i j , b j k -> b i k',
                c_enc_sa_rollout,
                c[enc_sa][:, l],
            )
            # c_enc_sa_rollout = self.normalize_contrib(c_enc_sa_rollout, norm_mode)
            c_roll[enc_sa].append(c_enc_sa_rollout.unsqueeze(1))
            
        dec_sa = 'decoder.self_attn'
        dec_ed = 'decoder.encoder_attn'
        print('c[dec_ed].size()',c[dec_ed].size())
        _, layers, t_out, t_in = c[dec_ed].size()

        c_dec_rollout = torch.eye(t_out).view(1, t_out, t_out, 1).to(c[dec_sa].device)
        for l in range(layers):
            c_dec_rollout = torch.einsum(
                'b i j e , b j k -> b i k e',
                c_dec_rollout,
                c[dec_sa][:, l],
            )
            
            c_dec_sa_rollout = c_dec_rollout.sum(-1)
            # c_dec_sa_rollout = self.normalize_contrib(c_dec_rollout.sum(-1), norm_mode)
            
            c_roll[dec_sa].append(c_dec_sa_rollout.unsqueeze(1))
            
            c_dec_rollout = torch.einsum(
                'b i j e , b j e , b e f -> b i j f',
                c_dec_rollout,
                c[dec_ed][:, l],
                c_enc_sa_rollout,
            )
            c_dec_ed_rollout = c_dec_rollout.sum(-3)
            # c_dec_ed_rollout = self.normalize_contrib(c_dec_rollout.sum(-3), norm_mode)
            
            c_roll[dec_ed].append(c_dec_ed_rollout.unsqueeze(1))

        return {k: torch.cat(v, dim=1) for k, v in c_roll.items()}
    
    def viz_contributions(self, src_tensor, tgt_tensor, contrib_type, roll=False, attn=None, layer=None, head=None, **contrib_kwargs):
        if roll:
            contrib = self.get_contribution_rollout(src_tensor, tgt_tensor, contrib_type, **contrib_kwargs)
        else:
            contrib = self.get_contributions(src_tensor, tgt_tensor, contrib_type, **contrib_kwargs)
        
        src_tok = self.decode(src_tensor, self.task.src_dict)
        tgt_tok = self.decode(tgt_tensor, self.task.tgt_dict)
        
        def what_to_show(arg, valid_values):
            valid_type = type(valid_values[0])
            if arg is None:
                to_show = valid_values
            elif isinstance(arg, valid_type):
                to_show = [arg]
            elif isinstance(arg, list):
                to_show = [a for a in arg if a in valid_values]
            else:
                raise TypeError("Argument must be str, List[str] or None")

            return to_show
        
        def show_contrib_heatmap(data, k_tok, q_tok, title):
            df = pd.DataFrame(
                data=data,
                columns=k_tok,
                index=q_tok
            )

            fig, ax = plt.subplots()
            g = sns.heatmap(df, cmap="Blues", cbar=True, square=True, ax=ax, fmt='.2f')
            g.set_title(title)
            g.set_xlabel("Key")
            g.set_ylabel("Query")
            g.set_xticklabels(g.get_xticklabels(), rotation=50, horizontalalignment='center',fontsize=10)
            g.set_yticklabels(g.get_yticklabels(),fontsize=10);

            fig.show()  

        for a in what_to_show(attn, self.ATTN_MODULES):
            enc_dec_, _, attn_module_ = self.parse_module_name(a)
            num_layers = self.get_module(enc_dec_).num_layers
            if a == 'encoder.self_attn':
                q_tok = src_tok + ['<EOS>']
                k_tok = src_tok + ['<EOS>']
            elif a == 'decoder.self_attn':
                q_tok = tgt_tok + ['<EOS>']
                k_tok = ['<EOS>'] + tgt_tok
            elif a == 'decoder.encoder_attn':
                q_tok = tgt_tok + ['<EOS>']
                k_tok = src_tok + ['<EOS>']
            else:
                pass
            
            #q_tok = (src_tok + ['<EOS>']) if a == 'encoder.self_attn' else (['<EOS>'] + tgt_tok)
            #k_tok = (['<EOS>'] + tgt_tok) if a == 'decoder.self_attn' else (src_tok + ['<EOS>'])

            for l in what_to_show(layer, list(range(num_layers))):
                num_heads = self.get_module(a.replace('.', f'.{l}.')).num_heads
                
                contrib_ = contrib[a][0,l]


                if contrib_type == 'attn_w' and roll == False:
                    for h in what_to_show(head, [-1] + list(range(num_heads))):
                        contrib__ = contrib_.mean(0) if h == -1 else contrib_[h]
                        show_contrib_heatmap(
                            contrib__.cpu().detach().numpy(), #3
                            k_tok,
                            q_tok,
                            title=f"{contrib_type}; {a}; layer: {l}; head: {'mean' if h == -1 else h}"
                        )
                else:
                    show_contrib_heatmap(
                        contrib_.cpu().detach().numpy(),
                        k_tok,
                        q_tok,
                        title=f"{contrib_type}; {a}; layer: {l}"
                    )

def parse_single_alignment(string, reverse=False, one_add=False, one_indexed=False):
    """
    Given an alignment (as a string such as "3-2" or "5p4"), return the index pair.
    """
    assert '-' in string or 'p' in string

    a, b = string.replace('p', '-').split('-')
    a, b = int(a), int(b)

    if one_indexed:
        a = a - 1
        b = b - 1
    
    if one_add:
        a = a + 1
        b = b + 1

    if reverse:
        a, b = b, a

    return a, b