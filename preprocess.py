from mosestokenizer import *
import sentencepiece as spm
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
iwslt14_dir = Path(os.environ['IWSLT14_DATA_DIR'])
gold_alignment_dir = Path(os.environ['GOLD_DATA_DIR']).as_posix()

Languages = ["de", "en"]
# Datas = ["train", "valid", "test"]
# progress_dir_path = '/home/usuaris/scratch/javier.ferrando/datasets/orig'
processed_dir_path = iwslt14_dir.as_posix()
print(processed_dir_path)
bpe_dir_path = '/home/usuaris/veu/javier.ferrando/norm-analysis-of-transformer/exp2_nmt/work/bpe_model_and_vocab/'

# lowercase gold alignment test data
for ln in Languages:
    with open(gold_alignment_dir + "/test.uc." + ln, encoding="utf-8") as fi, open(gold_alignment_dir + "/test." + ln, "w", encoding="utf-8") as fo:
        for line in fi:
            fo.write(line.lower())



# tokenize test data by moses tokenizer
print("tokenizing iwslt data by moses tokenizer...")
if not os.path.exists(processed_dir_path + '/tokenized'):
    os.mkdir(processed_dir_path + '/tokenized')
for ln in Languages:
    with open(processed_dir_path + '/preprocessed/test.' + ln, "r", encoding="utf-8") as fin:
        with open(processed_dir_path + '/tokenized/test.' + ln, "w", encoding="utf-8") as fout:
            with MosesTokenizer(ln) as tokenize:
                for sent in fin:
                    sent = sent.strip()
                    tokenized_sent = ' '.join(tokenize(sent)) + "\n"
                    fout.write(tokenized_sent)

print("tokenizing gold alignment by moses tokenizer...")
if not os.path.exists(gold_alignment_dir + '/tokenized'):
    os.mkdir(gold_alignment_dir + '/tokenized')
for ln in Languages:
    with open(gold_alignment_dir + "/test." + ln, "r", encoding="utf-8") as fin:
        with open(gold_alignment_dir + '/tokenized/test.' + ln, "w", encoding="utf-8") as fout:
            with MosesTokenizer(ln) as tokenize:
                for sent in fin:
                    sent = sent.strip()
                    tokenized_sent = ' '.join(tokenize(sent)) + "\n"
                    fout.write(tokenized_sent)



# tokenize each data by trained BPE models
print('tokenizing iwslt14 by trained BPE models...')
# if not os.path.exists(processed_dir_path + '/tokenized_bpe'):
#     os.mkdir(processed_dir_path + '/tokenized_bpe')
for lang in Languages:
    sp = spm.SentencePieceProcessor()
    sp.load(bpe_dir_path + lang + '.model')
    with open(processed_dir_path + '/tokenized/test.' + 'bpe.' + lang, 'w', encoding="utf-8") as fo:
        with open(processed_dir_path + '/tokenized/test.' + lang, encoding="utf-8") as fi:
            for line in fi:
                fo.write(" ".join(sp.EncodeAsPieces(line.strip())) + "\n")

# tokenize each data by trained BPE models
print('tokenizing gold alignment by trained BPE models...')
# if not os.path.exists(gold_alignment_dir + '/tokenized_bpe'):
#     os.mkdir(gold_alignment_dir + '/tokenized_bpe')
for lang in Languages:
    sp = spm.SentencePieceProcessor()
    sp.load(bpe_dir_path + lang + '.model')
    with open(gold_alignment_dir + '/tokenized/test.' + 'bpe.' + lang, 'w', encoding="utf-8") as fo:
        with open(gold_alignment_dir + '/tokenized/test.' + lang, encoding="utf-8") as fi:
            for line in fi:
                fo.write(" ".join(sp.EncodeAsPieces(line.strip())) + "\n")