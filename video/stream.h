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

    int comm_size;
    bool z_only_writer;
    MPI_Comm comm;

  public:

    /**
     * Initialize a video stream.
     * @param filename The name of the file to write to.
     * @param fps The number of frames per second.
     * @param size The size of the video final video.
     * @param workers_row The number of workers per row.
     * @param workers_col The number of workers per column.
     * @param z_only_writer If true, the rank is 0 is only the disk writer. If false, the rank 0 is a worker and the disk writer.
     * @param comm The MPI communicator : rank 0 is the disk writer, the other ranks are the workers which call append_frame.
     */
    Stream(const char* filename, int fps, cv::Size size, int workers_row, int workers_col, bool z_only_writer, MPI_Comm comm);

    /**
     * Destroy a video stream.
     */
    ~Stream();

    /**
     * Append a frame to the video stream.
     * If the rank is 0, the frame is written to the disk. If z_only_writer is true, the frame isn't taken into account.
     * If the rank is not 0, the frame is sent to the rank 0.
     * @param frame The frame to append.
     * @param frame_filename The name of the file to write to.
     */
    void append_frame(cv::Mat* frame, const char* frame_filename = nullptr);

    /**
     * Append a frame to the video stream (write the frame to the disk).
     * @param frame The frame to append.
     * @param frame_filename The name of the file to write to.
     */
    void append_frame(Eigen::Tensor<float, 3> &frame, const char* frame_filename = nullptr);


    /**
     * End the video stream.
     */
    void end_stream();
};

#endif // !VIDEO_H
