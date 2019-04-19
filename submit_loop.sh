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
#SBATCH --time=14:00:00
#SBATCH --mem=5000MB
#SBATCH --output=$(pwd)/Log/${OUTPUT}.log

cd \$SLURM_SUBMIT_DIR
./configure.sh Tidal3

start=\`date +%s\`
ID=0

while true
do
    INPUT=SkyrmeParameters/PowerLaw\${ID}.csv
    python GenerateMetaParameters.py
    mv test.csv \${INPUT}

    mpiexec -np ${NODES} --bind-to none --mca btl tcp,self --oversubscribe python -W ignore GenerateReport.py -i \${INPUT} -o ${OUTPUT}_\${ID} -c ${CORES} -pd 0.3 -pp 2.5 -et Meta2Poly --PBar -tg 1.1 1.2 1.3 1.4 1.5 1.7 1.8

    end=\`date +%s\`
    runtime=\$((end-start))
    if (( runtime > 46800 )); then
        break
    fi
    ID=\$((ID+1))
done

EOF
    sbatch --exclude=lac-040 $SUBFILE
    rm $SUBFILE
    #echo $SUBFILE
fi
