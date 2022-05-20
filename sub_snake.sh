#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --output=Snakemake.log
#SBATCH --mem=3G

#which python
snakemake -j 10 -p --config name='BillEOSWidePsymMoreMassPost' prior_name='BillEOSWidePsymMoreMass' --cluster-config cluster.json --cluster "sbatch --nodes=1 --mem-per-cpu={cluster.mem_per_cpu} --time={cluster.time} --output={cluster.output} --error={cluster.error} --ntasks={cluster.ntasks}" 
