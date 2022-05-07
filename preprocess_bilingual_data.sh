#!/bin/bash
#SBATCH -p veu             # Partition to submit to
#SBATCH --mem=16G      # Max CPU Memory
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

set -ex


export LC_ALL=en_US.UTF-8
export PATH=~/anaconda3/bin:$PATH
export EUROPARL_DATA_DIR=/home/usuaris/veu/javier.ferrando/norm-analysis-of-transformer/exp2_nmt/work

source activate int_nmt

export PYTHONUNBUFFERED=TRUE

export bpe_dir_path=/home/usuaris/veu/javier.ferrando/norm-analysis-of-transformer/exp2_nmt/work/bpe_model_and_vocab
src=de
tgt=en

for lang in en de
do
    python $HOME/fairseq/scripts/spm_encode.py \
        --model $bpe_dir_path/spm.128k.model \
        --output_format=piece \
        --inputs=$EUROPARL_DATA_DIR/processed_data/test.$lang \
        --outputs=$M2M_DATA_DIR/spm.$src.$tgt.$lang
done