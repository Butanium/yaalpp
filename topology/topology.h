#ifndef H_TOPOLOGY
#define H_TOPOLOGY

#include <mpi.h>

struct Topology {
  int nodes;
  int processes;
  int cores_per_process;
  int gpus;
};

Topology get_topology(MPI_Comm comm);

#endif // !H_TOPOLOGY
