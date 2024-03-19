#include "topology.h"
#include <mpi.h>
#include <omp.h>
#include <stdlib.h>
#include <cuda_runtime.h>

int get_nb_nodes(MPI_Comm comm) {
    int nb_nodes;
    int globalRank, localRank;
    MPI_Comm nodeComm, masterComm;
    MPI_Comm_rank(comm, &globalRank);
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, globalRank, MPI_INFO_NULL, &nodeComm);
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_split(comm, localRank, globalRank, &masterComm);
    MPI_Comm_free(&nodeComm);

    if (globalRank == 0) {
        MPI_Comm_size(masterComm, &nb_nodes);
    }

    MPI_Comm_free(&masterComm);

    MPI_Bcast(&nb_nodes, 1, MPI_INT, 0, comm);
    return nb_nodes;
}

int get_nb_processes(MPI_Comm comm) {
    int nb_processes;
    MPI_Comm_size(comm, &nb_processes);
    return nb_processes;
}

int get_nb_cores_per_process() {
    int nb_threads;
#pragma omp parallel
    {
#pragma omp master
        {
            nb_threads = omp_get_num_threads();
        }
    }
  return nb_threads;
}

int get_nb_gpus() {
    int nb_gpus;
    cudaGetDeviceCount(&nb_gpus);
    return nb_gpus;
}

int get_total_gpu_memory() {
    int nb_gpus;
    cudaGetDeviceCount(&nb_gpus);
    int total_memory = 0;
    for (int i = 0; i < nb_gpus; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        total_memory += prop.totalGlobalMem;
    }
    return total_memory;
}

Topology get_topology(MPI_Comm comm) {
    return {
       .nodes = get_nb_nodes(comm),
       .processes = get_nb_processes(comm),
       .cores_per_process = get_nb_cores_per_process(),
       .gpus = get_nb_gpus(),
       .gpu_memory = get_total_gpu_memory()
    };
}
