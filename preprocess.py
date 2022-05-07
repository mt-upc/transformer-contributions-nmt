from mosestokenizer import *
import sentencepiece as spm
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
iwslt14_dir = Path(os.environ['IWSLT14_DATA_DIR'])

Languages = ["de", "en"]
# Datas = ["train", "valid", "test"]
# progress_dir_path = '/home/usuaris/scratch/javier.ferrando/datasets/orig'
processed_dir_path = iwslt14_dir.as_posix()
print(processed_dir_path)
bpe_dir_path = '/home/usuaris/veu/javier.ferrando/norm-analysis-of-transformer/exp2_nmt/work/bpe_model_and_vocab/'


# # tokenize train data by moses tokenizer
# print("tokenizing by moses tokenizer...")
# for ln in Languages:
#     with open('./data/europarl-v7.de-en.' + ln, "r", encoding="utf-8") as fin:
#         with open(progress_dir_path + 'tokenized_train_valid.' + ln, "w", encoding="utf-8") as fout:
#             with MosesTokenizer(ln) as tokenize:
#                 for sent in fin:
#                     sent = sent.strip()
#                     tokenized_sent = ' '.join(tokenize(sent)) + "\n"
#                     fout.write(tokenized_sent)


# # lowercase test data
# for ln in Languages:
#     with open("./work/data_in_progress/test.uc." + ln, encoding="utf-8") as fi, open("./work/processed_data/test." + ln, "w", encoding="utf-8") as fo:
#         for line in fi:
#             fo.write(line.lower())

# # tokenize each data by trained BPE models
print('tokenizing by trained BPE models...')
for lang in Languages:
    sp = spm.SentencePieceProcessor()
    sp.load(bpe_dir_path + lang + '.model')
    with open(processed_dir_path + '/test' + '.bpe.' + lang, 'w', encoding="utf-8") as fo:
        with open(processed_dir_path + '/test' + '.' + lang, encoding="utf-8") as fi:
            for line in fi:
                fo.write(" ".join(sp.EncodeAsPieces(line.strip())) + "\n")