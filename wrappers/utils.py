import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import alignment.align as align

def spearmanr(x, y):
    """Compute Spearman rank's correlation bertween two attribution vectors.
        https://github.com/samiraabnar/attention_flow/blob/master/compute_corel_distilbert_sst.py"""

    x = pd.Series(x)
    y = pd.Series(y)
    assert x.shape == y.shape
    rx = x.rank(method='dense')
    ry = y.rank(method='dense')
    d = rx - ry
    dsq = np.sum(np.square(d))
    n = x.shape[0]
    coef = 1. - (6. * dsq) / (n * (n**2 - 1.))
    return [coef]

def get_normalized_rank(x):
    """Compute normalized [0,1] ranks. The higher the value, the higher the rank."""
    
    length_tok_sentence = x.shape
    x = pd.Series(x)
    rank = x.rank(method='dense')
    rank_normalized = rank/length_tok_sentence
    return rank_normalized

def visualize_alti(total_alti, source_sentence, target_sentence, predicted_sentence, word_level, alignment, all_layers=False):

    def plot_heatmap_alti(contributions_rollout_layer_np, source_sentence_,
                                  target_sentence_, predicted_sentence_):

        sns.set_style("whitegrid")
        fig = plt.figure(figsize=(17, 24), dpi=200)
        gs = GridSpec(3, 6)
        gs.update(wspace=1, hspace=0.05)
        ax_main = plt.subplot(gs[0:3, :5])
        ax_yDist = plt.subplot(gs[1, 5])
        
        df = pd.DataFrame(contributions_rollout_layer_np, columns = source_sentence_ + target_sentence_, index = predicted_sentence_)

        sns.set(font_scale=1.2)
        sns.heatmap(df,cmap="Blues",square=True,ax=ax_main,cbar=False)
        ax_main.axvline(x = len(source_sentence_)-0.02, lw=1.5, linestyle = '--', color = 'grey')
        ax_main.set_xlabel('Source sentence | Target prefix', fontsize=17)
        ax_main.set_ylabel('$\longleftarrow$ Decoding step', fontsize=17)
        ax_main.set_xticklabels(ax_main.get_xticklabels(), rotation=60)
        #ax_main.set_title('Layer ' + str(layer+1))

        src_contribution = contributions_rollout_layer_np[:, :len(source_sentence_)].sum(-1)
        df_src_contribution = pd.DataFrame(src_contribution, columns = ['src_contribution'], index = predicted_sentence_)

        ax_yDist.barh(range(0, len(predicted_sentence_)), df_src_contribution.src_contribution, align='center')
        plt.yticks(ticks = range(0, len(predicted_sentence_)) ,labels = predicted_sentence_,fontsize='14')
        plt.gca().invert_yaxis()
        ax_yDist.grid(True, linestyle=(0, (5, 10)));
        ax_yDist.set_xlim(0,1)
        ax_yDist.spines['top'].set_visible(False)
        ax_yDist.spines['right'].set_visible(False)
        ax_yDist.spines['bottom'].set_visible(False)
        ax_yDist.spines['left'].set_visible(False)
        ax_yDist.xaxis.set_ticks_position("bottom")
        ax_yDist.set_title('Source contribution')
    
    if all_layers:

        for layer in range(0, total_alti.shape[0]):

            contributions_rollout_layer = total_alti[layer]
            contributions_rollout_layer_np = contributions_rollout_layer.detach().cpu().numpy()
                
            if word_level:
                if alignment:
                    tokens_out = target_sentence[1:] + ['</s>']
                else:
                    tokens_out = predicted_sentence
                contributions_rollout_layer_np, words_in, words_out = align.contrib_tok2words(
                    contributions_rollout_layer_np,
                    tokens_in=(source_sentence + target_sentence),
                    tokens_out=tokens_out
                )
            source_sentence_ = words_in[:words_in.index('</s>')+1] if word_level else source_sentence
            target_sentence_ = words_in[words_in.index('</s>')+1:] if word_level else target_sentence
            predicted_sentence_ = words_out if word_level else predicted_sentence


            plot_heatmap_alti(contributions_rollout_layer_np, source_sentence_,
                                target_sentence_, predicted_sentence_)

    else:
        layer = -1
        contributions_rollout_layer = total_alti[layer]
        contributions_rollout_layer_np = contributions_rollout_layer.detach().cpu().numpy()
            
        if word_level:
            if alignment:
                tokens_out = target_sentence[1:] + ['</s>']
            else:
                tokens_out = predicted_sentence
            contributions_rollout_layer_np, words_in, words_out = align.contrib_tok2words(
                contributions_rollout_layer_np,
                tokens_in=(source_sentence + target_sentence),
                tokens_out=tokens_out
            )
        source_sentence_ = words_in[:words_in.index('</s>')+1] if word_level else source_sentence
        target_sentence_ = words_in[words_in.index('</s>')+1:] if word_level else target_sentence
        predicted_sentence_ = words_out if word_level else predicted_sentence

        plot_heatmap_alti(contributions_rollout_layer_np, source_sentence_,
                            target_sentence_, predicted_sentence_)

    return contributions_rollout_layer_np, source_sentence_, predicted_sentence_
    