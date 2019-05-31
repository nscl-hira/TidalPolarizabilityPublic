if [ $# -ne 3 ]; then
    echo 'To use this script, you need'
    echo $0 ' NumNodes NumCoresPerNodes OutputFileName'
else
    NODES=$1
    CORES=$2
    OUTPUT=$3
    SUBFILE=$(mktemp /tmp/GenerateReportSubmit.XXXXXX)

# remember to disable infiniband because it is incompetable with fork in openmpi
cat > $SUBFILE << EOF
#!/bin/sh
#SBATCH --cpus-per-task=${CORES}
#SBATCH --nodes=${NODES}
#SBATCH --time=12:00:00
#SBATCH --mem=5000MB
#SBATCH --output=$(pwd)/Log/${OUTPUT}.log


export OMPI_MCA_btl_openib_allow_ib=1
cd \$SLURM_SUBMIT_DIR
module load GCC/8.2.0-2.31.1
module load OpenMPI/4.0.0
export PATH="/mnt/home/tsangchu/anaconda2/bin:\$PATH"
source activate Tidal3

which python
conda deactivate 
conda activate Tidal3

which python
echo "starting mpi"
mpiexec -np $(( NODES*CORES )) --oversubscribe python -W ignore MakeSkyrmeFileBisection.py -o ${OUTPUT}


EOF
    sbatch --exclude=lac-040 $SUBFILE
    rm $SUBFILE
    #echo $SUBFILE
fi
