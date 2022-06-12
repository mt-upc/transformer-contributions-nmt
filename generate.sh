#!/bin/bash
#SBATCH -p veu             # Partition to submit to
#SBATCH --mem=16G      # Max CPU Memory
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -ex


export LC_ALL=en_US.UTF-8
export PATH=~/anaconda3/bin:$PATH
export M2M_CKPT_DIR=/home/usuaris/scratch/javier.ferrando/checkpoints/m2m_100
export M2M_DATA_DIR=/home/usuaris/scratch/javier.ferrando/datasets/m2m_data
export EUROPARL_DATA_DIR=/home/usuaris/veu/javier.ferrando/norm-analysis-of-transformer/exp2_nmt/work
export IWSLT14_DATA_DIR=/home/usuaris/scratch/javier.ferrando/datasets/iwslt14

source activate int_nmt

export PYTHONUNBUFFERED=TRUE

src=en
tgt=es

fairseq-generate $M2M_DATA_DIR/flores101_dataset/data_bin --batch-size 1 --path $M2M_CKPT_DIR/418M_last_checkpoint.pt --fixed-dictionary $M2M_CKPT_DIR/model_dict.128k.txt -s $src -t $tgt --remove-bpe 'sentencepiece' --beam 5 --task translation_multi_simple_epoch --lang-pairs $M2M_CKPT_DIR/language_pairs_small_models.txt --decoder-langtok --encoder-langtok src --gen-subset test > gen_out