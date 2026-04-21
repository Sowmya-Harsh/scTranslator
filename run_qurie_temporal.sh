#!/bin/bash -l
#SBATCH --job-name=temporal_proteo
#SBATCH --output=logs/temporal_proteo_%j.out
#SBATCH --error=logs/temporal_proteo_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

module --force purge
module load env/legacy/2020b
module load lang/Anaconda3/2020.11

cd /mnt/aiongpfs/users/sjanmahanthi/Temporal_PTM_scTranslator/scTranslator

mkdir -p logs
mkdir -p checkpoint/qurie_temporal

PYTHON=~/.conda/envs/performer/bin/python

echo "=============================="
echo " TemporalProteoTranslator"
echo " Job ID: $SLURM_JOB_ID"
echo " GPU: $CUDA_VISIBLE_DEVICES"
echo " Node: $SLURMD_NODENAME"
echo "=============================="

# ── Run 1: with delta-t (full model) ──────────────────────────────────────
echo ""
echo "--- Run 1: with delta-t conditioning ---"
$PYTHON train_qurie_temporal.py \
    --data_dir /mnt/aiongpfs/users/sjanmahanthi/Data_Test/QuRIE_processed \
    --pretrain_checkpoint checkpoint/scTranslator_2M.pt \
    --output_dir checkpoint/qurie_temporal \
    --epochs 150 \
    --batch_size 8 \
    --lr 5e-6 \
    --val_fraction 0.1 \
    --lambda_ptm 0.5 \
    --lambda_ratio 0.3 \
    --freeze_epochs 150 \
    --patience 15 \
    --accum_steps 4 \
    --detail_every 25 \
    --max_pairs 5000

# ── Run 2: without delta-t (ablation) ─────────────────────────────────────
echo ""
echo "--- Run 2: ablation — no delta-t ---"
$PYTHON train_qurie_temporal.py \
    --data_dir /mnt/aiongpfs/users/sjanmahanthi/Data_Test/QuRIE_processed \
    --pretrain_checkpoint checkpoint/scTranslator_2M.pt \
    --output_dir checkpoint/qurie_temporal \
    --epochs 150 \
    --batch_size 8 \
    --lr 5e-6 \
    --val_fraction 0.1 \
    --lambda_ptm 0.5 \
    --lambda_ratio 0.3 \
    --freeze_epochs 150 \
    --patience 15 \
    --accum_steps 4 \
    --detail_every 25 \
    --max_pairs 5000 \
    --no_delta_t

echo ""
echo "=============================="
echo " Done. Results in checkpoint/qurie_temporal/"
echo "=============================="
