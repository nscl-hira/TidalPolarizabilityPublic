#!/bin/sh
#SBATCH --cpus-per-task=20
#SBATCH --nodes=20
#SBATCH --time=01:00:00
#SBATCH --mem=5000MB
#SBATCH --output=test_srun.out

cd $SLURM_SUBMIT_DIR
echo $SLURM_SUBMIT_DIR

module unload python
export PATH="/mnt/home/tsangchu/anaconda2/bin:$PATH"
source activate Tidal
mpiexec -np 20 python -W ignore GenerateReport.py -i SkyrmeParameters/test.csv -o test -pd 0.3 -et EOS --PBar

