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

source activate int_nmt

export PYTHONUNBUFFERED=TRUE

src=de
tgt=en

for lang in en de
do
    python $HOME/fairseq/scripts/spm_encode.py \
        --model $M2M_CKPT_DIR/spm.128k.model \
        --output_format=piece \
        --inputs=$EUROPARL_DATA_DIR/processed_data/test.$lang \
        --outputs=$M2M_DATA_DIR/spm.$src.$tgt.$lang
done

cd M2M_DATA_DIR

pwd

fairseq-preprocess \
    --source-lang $src --target-lang $tgt \
    --testpref $M2M_DATA_DIR/spm.$src.$tgt \
    --thresholdsrc 0 --thresholdtgt 0 \
    --destdir $M2M_DATA_DIR/data_bin \
    --srcdict $M2M_CKPT_DIR/data_dict.128k.txt --tgtdict $M2M_CKPT_DIR/data_dict.128k.txt