#include "topology.h"
#include <mpi.h>
#include <stdlib.h>


int get_nb_nodes(MPI_Comm comm) {
  int nb_nodes;
  int globalRank, localRank;
  MPI_Comm nodeComm, masterComm;

  MPI_Comm_rank(comm, &globalRank);
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, globalRank, MPI_INFO_NULL, &nodeComm);
  MPI_Comm_rank(nodeComm, &localRank);
  MPI_Comm_split(comm, localRank, globalRank, &masterComm);
  MPI_Comm_free(&nodeComm);

  if ( globalRank == 0 ) {
    MPI_Comm_size(masterComm, &nb_nodes);
  }

  MPI_Comm_free( &masterComm );

  MPI_Bcast(&nb_nodes, 1, MPI_INT, 0, comm);
  return nb_nodes;
}

int get_nb_processes(MPI_Comm comm) {
  int nb_processes;
  MPI_Comm_size(comm, &nb_processes);
  return nb_processes;
}

int get_nb_cores_per_process() {
  int nb_threads = atoi(getenv("OMP_NUM_THREADS"));
  return nb_threads;
}

Topology get_topology(MPI_Comm comm) {
  Topology t;
  t.nodes = get_nb_nodes(comm);
  t.processes = get_nb_processes(comm);
  t.cores_per_process = get_nb_cores_per_process();
  return t;
}
