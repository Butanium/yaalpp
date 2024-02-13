#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/reporters/catch_reporter_console.hpp>
#include <catch2/reporters/catch_reporter_helpers.hpp>

#include <mpi.h>
#include "../video/stream.h"
#include <unsupported/Eigen/CXX11/Tensor>

unsigned int Factorial( unsigned int number ) {
    return number <= 1 ? number : Factorial(number-1)*number;
}

TEST_CASE( "Factorials are computed", "[factorial]" ) {
    REQUIRE( Factorial(1) == 1 );
    REQUIRE( Factorial(2) == 2 );
    REQUIRE( Factorial(3) == 6 );
    REQUIRE( Factorial(10) == 3628800 );
}

TEST_CASE( "Output video multiple process", "[output_multiple]" ) {
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if(comm_size != 5) {
      return;
    }

    Stream stream("output_multiple_mpi.mp4", cv::Size(1000, 1000), 2, 2, MPI_COMM_WORLD);
    
    int rank_id;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank_id);
    if(rank_id == 0) {
      for (int i = 0; i < 16; i++) stream.write_frame();
    } else {
      Eigen::Tensor<float, 3> map(2,2,3);

      for (int i = 0; i < 16; i++) {
        map.setZero();

        if ((i/4)+1 == rank_id) {
          int x = i%4;
          map(x%2, x/2, 0) = 1;
          map(x%2, x/2, 1) = 1;
          map(x%2, x/2, 2) = 1;
        }

        stream.append_frame(map);
      }
    }
    stream.end_stream();
}

TEST_CASE( "Output video one process", "[output_single]" ) {
    int comm_size;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    if(comm_size != 1) {
      return;
    }

    Stream stream("output_single_mpi.mp4", cv::Size(1000, 1000), 1, 1, MPI_COMM_WORLD);
    Eigen::Tensor<float, 3> map(3,3,3);

    for (int i = 0; i < 10; i++) {
      map.setZero();

      map(i/3, i%3, 0) = 1;
      map(i/3, i%3, 1) = 1;
      map(i/3, i%3, 2) = 1;

      stream.append_frame(map);
    }
    stream.end_stream();
}

// Custom main function to handle MPI initialization and finalization
int main( int argc, char* argv[] ) {
    MPI_Init(&argc, &argv);
    int result = Catch::Session().run( argc, argv );
    MPI_Finalize();
    return result;
}
