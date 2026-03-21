def pytest_addoption(parser):
    parser.addoption(
        "--target", 
        action="store", 
        default="./serial/src/docs",
        help="Path to the C++ binary to test"
    )
    parser.addoption(
        "--mpi-procs",
        type=int,
        default=0,
        help="Number of MPI processes (0 = no MPI)"
    )
    parser.addoption(
        "--mpi-launcher",
        action="store",
        default="mpirun",
        help="MPI launcher (mpirun or srun)"
    )