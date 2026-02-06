#!/bin/ksh 
#$ -q gpu
#$ -o result.out
#$ -j y
#$ -N diffusion_schrodinger_bridge
cd $WORKDIR
cd /beegfs/data/work/imvia/in156281/diffusion_schrodinger_bridge
source /beegfs/data/work/imvia/in156281/diffusion_schrodinger_bridge/venv/bin/activate
module load python
export PYTHONPATH=/work/imvia/in156281/diffusion_schrodinger_bridge/venv/lib/python3.9/site-packages:$PYTHONPATH
export MPLCONFIGDIR=/work/imvia/in156281/.cache/matplotlib

python train.py