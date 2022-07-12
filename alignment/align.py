import numpy as np


def contrib_tok2words_partial(contributions, tokens, axis, reduction):
    from string import punctuation

    reduction_fs = {
        'avg': np.mean,
        'sum': np.sum
    }

    words = []
    w_contributions = []
    for counter, (tok, contrib) in enumerate(zip(tokens, contributions.T)):
        if tok.startswith('▁') or tok.startswith('__') or tok.startswith('<') or counter==0:# or tok in punctuation:
            if tok.startswith('▁'):
                tok = tok[1:]
            words.append(tok)
            w_contributions.append([contrib])
        else:
            words[-1] += tok
            w_contributions[-1].append(contrib)

    reduction_f = reduction_fs[reduction]
    word_contrib = np.stack([reduction_f(np.stack(contrib, axis=axis), axis=axis) for contrib in w_contributions], axis=axis)

    return word_contrib, words


def contrib_tok2words(contributions, tokens_in, tokens_out):
    word_contrib, words_in = contrib_tok2words_partial(contributions, tokens_in, axis=0, reduction='sum')
    word_contrib, words_out = contrib_tok2words_partial(word_contrib, tokens_out, axis=1, reduction='avg')
    return word_contrib.T, words_in, words_out


def get_word_word_attention(token_token_attention, src_word_to_bpe, trg_word_to_bpe, remove_EOS=True):
    '''
    From bpe tokens-tokens attention to words-words attention
    '''
    #src_word_to_bpe -> [[0],[1],[2,3,4]]...]
    word_word_attention = np.array(token_token_attention)
    not_word_starts = []
    for word in src_word_to_bpe:
        not_word_starts += word[1:]

    # sum up the contributions for all tokens in a source word that has been split
    for word in src_word_to_bpe:
        word_word_attention[:, word[0]] = word_word_attention[:, word].sum(axis=-1)
    word_word_attention = np.delete(word_word_attention, not_word_starts, -1)

    not_word_starts = []
    for word in trg_word_to_bpe:
        not_word_starts += word[1:]
    
    # mean the contributions for all tokens in a target word that has been split
    # other merging alternatives may work too
    for word in trg_word_to_bpe:
        word_word_attention[word[0]] = np.mean(word_word_attention[word], axis=0)
        
    word_word_attention = np.delete(word_word_attention, not_word_starts, 0)
    if remove_EOS:
        word_word_attention = np.delete(word_word_attention, -1, 0)

    return word_word_attention

def convert_bpe_word(splited_bpe_sent, splited_word_sent):
    """
    Given a sentence made of BPE subwords and words (as a string), 
    returns a list of lists where each sub-list contains BPE subwords
    for the correponding word.
    """

    #splited_bpe_sent = bpe_sent.split()
    #splited_word_sent = splited_word_sent.split()
    word_to_bpe = [[] for _ in range(len(splited_word_sent))]

    
    word_i = 0
    for bpe_i, token in enumerate(splited_bpe_sent):
        # if bpe_i == 0:
        #     # First token may use ▁
        #     word_to_bpe[word_i].append(bpe_i)
        # else:
        if token.startswith("▁"):
            word_i += 1
            
        word_to_bpe[word_i].append(bpe_i)
    
    for word in word_to_bpe:
        assert len(word) != 0
    
    word_to_bpe.append([len(splited_bpe_sent)])
    return word_to_bpe

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