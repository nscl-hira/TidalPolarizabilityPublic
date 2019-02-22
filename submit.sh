if [ $# -ne 4 ]; then
    echo 'To use this script, you need'
    echo $0 ' NumNodes NumCoresPerNodes InputParName OutputFileName'
else
    NODES=$1
    CORES=$2
    INPUT=$3
    OUTPUT=$4
    SUBFILE=$(mktemp /tmp/GenerateReportSubmit.XXXXXX)

cat > $SUBFILE << EOF
#!/bin/sh
#SBATCH --cpus-per-task=${CORES}
#SBATCH --nodes=${NODES}
#SBATCH --time=05:00:00
#SBATCH --mem=5000MB
#SBATCH --output=$(pwd)/Log/${OUTPUT}.log

cd \$SLURM_SUBMIT_DIR
./configure.sh Tidal3
mpiexec -np ${NODES} --bind-to none python -W ignore GenerateReport.py -i ${INPUT} -o ${OUTPUT} -pd 0.3 -et EOS --PBar 
EOF
    sbatch $SUBFILE
    rm $SUBFILE
fi
