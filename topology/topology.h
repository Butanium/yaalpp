#pragma once

#include <mpi.h>

struct Topology {
  int nodes;
  int processes;
  int cores_per_process;
  int gpus;
  int gpu_memory;
};

Topology get_topology(MPI_Comm comm);

