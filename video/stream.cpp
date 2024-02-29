#include "stream.h"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include <mpi.h>

using namespace std;
using namespace cv;
using Eigen::Tensor;


/**
 * Convert an Eigen Tensor to an OpenCV Mat.
 * @param tensor The tensor to convert.
 * @return A pointer to the image matrix.
 */
Mat* eigen2cv(Tensor<float, 3> &tensor) {
  Mat* image = new Mat(tensor.dimension(0), tensor.dimension(1), CV_8UC3);

  for (int i = 0; i < tensor.dimension(0); i++) {
    for (int j = 0; j < tensor.dimension(1); j++) {
      BGR* pixel = image->ptr<BGR>(i, j);
      pixel->red = (unsigned char)(tensor(i, j, 0) * 255);
      pixel->green = (unsigned char)(tensor(i, j, 1) * 255);
      pixel->blue = (unsigned char)(tensor(i, j, 2) * 255);
    }
  }

  return image;
}

Stream::Stream(const char* filename, int fps, Size size, int workers_row, int workers_col, bool z_only_writer, MPI_Comm comm)
    : filename(filename), size(size), workers_row(workers_row), workers_col(workers_col), z_only_writer(z_only_writer), comm(comm) {
  int rank, comm_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_size);

  if (comm_size == 1 && z_only_writer) {
    cerr << "There is no worker" << endl;
    MPI_Abort(comm, 1);
  }

  if (comm_size != workers_row * workers_col + (z_only_writer ? 1 : 0)) {
    if (rank == 0) {
      cerr << "The number of workers is not equal to workers_row * workers_col" << endl;
    }
    MPI_Abort(comm, 1);
  }

  this->comm_size = comm_size;

  Size worker_size(size.width / workers_col, size.height / workers_row);
  this->worker_size = worker_size;

  if (rank == 0) { // This is the disk writer
    this->writer = new VideoWriter();
    this->img_buffer = new Mat(size, CV_8UC3);
    this->buffer = new uchar[this->img_buffer->total() * this->img_buffer->elemSize()];

    this->recvcounts = new int[comm_size];
    this->displs = new int[comm_size];

    if (z_only_writer) {
      this->recvcounts[0] = 0;
      this->displs[0] = 0;
    } else {
      this->recvcounts[0] = worker_size.area() * this->img_buffer->elemSize();
      this->displs[0] = 0;
    }

    for (int i = 1; i < comm_size; i++) {
      this->recvcounts[i] = worker_size.area() * this->img_buffer->elemSize();
      this->displs[i] = this->displs[i - 1] + this->recvcounts[i - 1];
    }

    int codec = VideoWriter::fourcc('a', 'v', 'c', '1');

    this->writer->open(filename, codec, fps, size, true);
  } else { // This is a worker
    this->writer = nullptr;
    this->img_buffer = nullptr;
    this->buffer = nullptr;

    this->recvcounts = nullptr;
    this->displs = nullptr;
  }
}

Stream::~Stream() {
  if (this->writer != nullptr) {
    delete this->writer;
    delete this->img_buffer;
    delete this->buffer;
  }
}

void Stream::append_frame(Mat* frame, const char* filename) {
  Mat resized;

  if (this->writer == nullptr || !this->z_only_writer) {
    resize(*frame, resized, this->worker_size, 0, 0, INTER_NEAREST);
  }

  MPI_Gatherv(resized.data, resized.total() * resized.elemSize(), MPI_UNSIGNED_CHAR, this->buffer, this->recvcounts, this->displs, MPI_UNSIGNED_CHAR, 0, this->comm);

  if (this->writer != nullptr) {

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < comm_size - (this->z_only_writer ? 1 : 0) ; i++) {
      Mat worker_frame(this->worker_size, CV_8UC3, this->buffer + (i * this->worker_size.area() * this->img_buffer->elemSize()));
      Rect worker_rect( (i % this->workers_row) * this->worker_size.width, (i / this->workers_row) * this->worker_size.height, this->worker_size.width, this->worker_size.height);

      worker_frame.copyTo( (*this->img_buffer)(worker_rect) );
    }

    this->writer->write(*this->img_buffer);

    if (filename != nullptr) {
      imwrite(filename, *this->img_buffer);
    }
  }
}

void Stream::append_frame(Tensor<float, 3> &frame, const char* filename) {
  Mat* image = eigen2cv(frame);
  this->append_frame(image);
  delete image;
}

void Stream::end_stream() {
  if (this->writer == nullptr) {
    return;
  }

  this->writer->release();
  cout << "Video written to " << this->filename << endl;
}
