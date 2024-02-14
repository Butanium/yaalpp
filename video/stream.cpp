#include "stream.h"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include <mpi.h>

#define FPS 2

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

Stream::Stream(const char* filename, Size size, int workers_row, int workers_col, MPI_Comm comm) : filename(filename), size(size), workers_row(workers_row), workers_col(workers_col), comm(comm) {
  int rank, comm_size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &comm_size);

  if (comm_size != workers_row * workers_col) {
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

    int codec = VideoWriter::fourcc('a', 'v', 'c', '1');

    this->writer->open(filename, codec, FPS, size, true);
  } else { // This is a worker
    this->writer = nullptr;
    this->img_buffer = nullptr;
    this->buffer = nullptr;
  }
}

Stream::~Stream() {
  if (this->writer != nullptr) {
    delete this->writer;
    delete this->img_buffer;
    delete this->buffer;
  }
}

void Stream::append_frame(Mat* frame) {
  Mat resized;
  resize(*frame, resized, this->worker_size, 0, 0, INTER_NEAREST);

  MPI_Gather(resized.data, resized.total() * resized.elemSize(), MPI_UNSIGNED_CHAR, this->buffer, resized.total() * resized.elemSize(), MPI_UNSIGNED_CHAR, 0, this->comm);

  if (this->writer != nullptr) {
    for (int i = 0; i < comm_size; i++) {
      Mat worker_frame(this->worker_size, CV_8UC3, this->buffer + (i * resized.total() * resized.elemSize()));
      Rect worker_rect( (i % this->workers_row) * this->worker_size.width, (i / this->workers_row) * this->worker_size.height, this->worker_size.width, this->worker_size.height);

      worker_frame.copyTo( (*this->img_buffer)(worker_rect) );
    }

    this->writer->write(*this->img_buffer);
  }
}

void Stream::append_frame(Tensor<float, 3> &frame) {
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
