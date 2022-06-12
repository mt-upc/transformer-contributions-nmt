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

SRC_LANG_CODE=eng
TRG_LANG_CODE=spa

SRC_MM100_LANG_CODE=en
TRG_MM100_LANG_CODE=es

mkdir -p $M2M_DATA_DIR/flores101_dataset/tokenized

# for lang in eng spa
# do
python $HOME/fairseq/scripts/spm_encode.py \
    --model $M2M_CKPT_DIR/spm.128k.model \
    --output_format=piece \
    --inputs=$M2M_DATA_DIR/flores101_dataset/devtestfake/${SRC_LANG_CODE}.devtest \
    --outputs=$M2M_DATA_DIR/flores101_dataset/tokenized/spm.${SRC_MM100_LANG_CODE}-${TRG_MM100_LANG_CODE}.${SRC_MM100_LANG_CODE}

python $HOME/fairseq/scripts/spm_encode.py \
    --model $M2M_CKPT_DIR/spm.128k.model \
    --output_format=piece \
    --inputs=$M2M_DATA_DIR/flores101_dataset/devtestfake/${TRG_LANG_CODE}.devtest \
    --outputs=$M2M_DATA_DIR/flores101_dataset/tokenized/spm.${SRC_MM100_LANG_CODE}-${TRG_MM100_LANG_CODE}.${TRG_MM100_LANG_CODE}
# done

#--outputs=/home/usuaris/veu/javier.ferrando/transformer-contributions-nmt/data/flores/test.spm.$lang
#--inputs=$IWSLT14_DATA_DIR/untokenized/test.$lang \
#--outputs=$IWSLT14_DATA_DIR/tokenized_m2m/test.spm.$lang

# --inputs=$EUROPARL_DATA_DIR/processed_data/test.$lang \
# --outputs=$M2M_DATA_DIR/spm.$src.$tgt.$lang

# cd M2M_DATA_DIR

# pwd
mkdir -p $M2M_DATA_DIR/flores101_dataset/data_bin

fairseq-preprocess \
    --source-lang ${SRC_MM100_LANG_CODE} --target-lang ${TRG_MM100_LANG_CODE} \
    --testpref $M2M_DATA_DIR/flores101_dataset/tokenized/spm.${SRC_MM100_LANG_CODE}-${TRG_MM100_LANG_CODE} \
    --thresholdsrc 0 --thresholdtgt 0 \
    --destdir $M2M_DATA_DIR/flores101_dataset/data_bin \
    --srcdict $M2M_CKPT_DIR/data_dict.128k.txt --tgtdict $M2M_CKPT_DIR/data_dict.128k.txt