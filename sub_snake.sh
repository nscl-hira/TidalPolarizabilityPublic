#!/bin/sh
#SBATCH --time=20:00:00
#SBATCH --output=Snakemake.log

module unload Python
export OMPI_MCA_btl_openib_allow_ib=1
module load GNU/8.2.0-2.31.1
module load OpenMPI/4.0.0

conda activate Tidal3

snakemake -j 1 --cluster-config cluster.json --cluster "sbatch --mem={cluster.mem} --time={cluster.time} --output={cluster.output} --error={cluster.error} --ntasks={cluster.ntasks} --exclude=lac-040" -p --config name='FullTest2'
