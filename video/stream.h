#ifndef VIDEO_H 
#define VIDEO_H

#include <opencv2/opencv.hpp>
#include <unsupported/Eigen/CXX11/Tensor>
#include <mpi.h>

struct BGR {
  unsigned char blue;
  unsigned char green;
  unsigned char red;
};

class Stream {
  private:
    const char* filename;
    cv::VideoWriter* writer;
    cv::Size size;
    cv::Size worker_size;
    int workers_row;
    int workers_col;

    uchar* buffer;
    cv::Mat* img_buffer;

    int* recvcounts;
    int* displs;


    size_t worker_img_size;
    MPI_Comm comm;

  public:

    /**
     * Initialize a video stream.
     * @param filename The name of the file to write to.
     * @param size The size of the video final video.
     * @param workers_row The number of workers per row.
     * @param workers_col The number of workers per column.
     * @param comm The MPI communicator : rank 0 is the disk writer, the other ranks are the workers which call append_frame.
     */
    Stream(const char* filename, cv::Size size, int workers_row, int workers_col, MPI_Comm comm);

    /**
     * Destroy a video stream.
     */
    ~Stream();

    /**
     * Append a frame to the video stream (write the frame to the disk).
     * @param frame The frame to append.
     */
    void append_frame(cv::Mat* frame);

    /**
     * Append a frame to the video stream (write the frame to the disk).
     * @param frame The frame to append.
     */
    void append_frame(Eigen::Tensor<float, 3> &frame);


    /**
     * Write a frame to the video stream. Used only by the disk writer in a multi-node setup, useless otherwise.
     */
    void write_frame();

    /**
     * End the video stream.
     */
    void end_stream();
};

#endif // !VIDEO_H
