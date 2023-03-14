#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --output=Snakemake.log
#SBATCH --mem=3G

#which python
snakemake -k -j 10 -p --config name='MetaM4Prior' prior_name='MetaM4Prior' --cluster-config cluster.json --cluster "sbatch --nodes=1 --mem-per-cpu={cluster.mem_per_cpu} --time={cluster.time} --output={cluster.output} --error={cluster.error} --ntasks={cluster.ntasks}" 
