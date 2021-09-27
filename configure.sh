if [ "$#" -ne 1 ]; then
    echo "Please supply the name of the environment"
else
    # check if mpi4py is installed
    # need to install with OpenMPI 4.0
    # cannot rely on conda bundle which uses mpich
    pip show mpi4py 1>/dev/null 
    if [ $? == 0 ]; then
       echo "Installed" #Replace with your actions
    else
        MPICC_DIR=$(which mpicc)
        git clone https://github.com/mpi4py/mpi4py.git ./mpi4py.git
        cd mpi4py.git
        python setup.py build --mpicc=${MPICC_DIR}
        python setup.py install
        cd ../
        rm -rf mpi4py.git
    fi
fi
