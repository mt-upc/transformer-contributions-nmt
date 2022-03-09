import numpy as np

def top_percentage(data, percentage=10):
    data = np.sort(data, axis=?)
    n = data.shape[0]
    threshold_idx = n*percentage/100
    return data[threshold_idx]

def top_k_words(data, k=5):
    data = np.sort(data, axis=?)
    return data[k-1]


def check_threshold(data, method):
    if method == 'top_percentage':
        threshold = top_percentage(data)
        if prob >= threshold:
            return True
        return False

    elif method == 'beam_search':
        threshold = top_k_words(data)
        if prob >= threshold:
            return True
        return False

def weighted_dot_product(weight, out, gold):
    return weight*(np.dot(out, gold))

def sentence_score(C_enc_all, C_dec_all, encdec_all, encdec_gold_all, selfdec_all, selfdec_gold_all):
    total_sum = 0
    n = len(C_enc_all)
    for C_enc, C_dec, encdec, encdec_gold, selfdec, selfdec_gold in \
         zip(C_enc_all, C_dec_all, encdec_all, encdec_gold_all, selfdec_all, selfdec_gold_all):

        if not check_threshold(enc_prob, 'top_percentage'):
            n = n-1
            continue
        if not check_threshold(dec_prob, 'top_percentage'):
            n = n-1
            continue
        
        total_sum = total_sum + weighted_dot_product(C_enc, encdec, encdec_gold) + weighted_dot_product(C_dec, selfdec, selfdec_gold)

    return total_sum/n

if __name__ == '__main__':
    sentence_score(C_enc_all, C_dec_all, encdec_all, encdec_gold_all, selfdec_all, selfdec_gold_all)




############################## OLD CODE ##############################

# def decoder_threshold(method, alpha=0.05):
#     if method == 'top_percentage':
#         threshold = top_percentage(data)
#         if prob > threshold:
#             return True
#         return False

#     elif method == 'beam_search':
#         threshold = top_k_words(data)
#         if prob > threshold:
#             return True
#         return False