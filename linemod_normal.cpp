/****************************************************************************************\
*                               Depth normal modality                                    *
\****************************************************************************************/
#include <iostream>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
// Contains GRANULARITY and NORMAL_LUT
#include "normal_lut.i"

#include "surface_matching/ppf_helpers.hpp"

using namespace cv;
using namespace ppf_match_3d;

static void accumBilateral(long delta, long i, long j, long * A, long * b, int threshold)
{
  long f = std::abs(delta) < threshold ? 1 : 0;

  const long fi = f * i;
  const long fj = f * j;

  A[0] += fi * i;
  A[1] += fi * j;
  A[3] += fj * j;
  b[0]  += fi * delta;
  b[1]  += fj * delta;
}

/**
 * \brief Compute quantized normal image from depth image.
 *
 * Implements section 2.6 "Extension to Dense Depth Sensors."
 *
 * \param[in]  src  The source 16-bit depth image (in mm).
 * \param[out] dst  The destination 8-bit image. Each bit represents one bin of
 *                  the view cone.
 * \param distance_threshold   Ignore pixels beyond this distance.
 * \param difference_threshold When computing normals, ignore contributions of pixels whose
 *                             depth difference with the central pixel is above this threshold.
 *
 * \todo Should also need camera model, or at least focal lengths? Replace distance_threshold with mask?
 */
static void quantizedNormals(const Mat& src, Mat& dst, Mat& normal, int distance_threshold,
                      int difference_threshold, const float cam[4])
{
  dst = Mat::zeros(src.size(), CV_8U);
  normal = Mat::zeros(src.cols*src.rows, 6, CV_32FC1);

  const unsigned short * lp_depth   = src.ptr<ushort>();
  unsigned char  * lp_normals = dst.ptr<uchar>();

  const int l_W = src.cols;
  const int l_H = src.rows;

  const int l_r = 5; // used to be 7
  const int l_offset0 = -l_r - l_r * l_W;
  const int l_offset1 =    0 - l_r * l_W;
  const int l_offset2 = +l_r - l_r * l_W;
  const int l_offset3 = -l_r;
  const int l_offset4 = +l_r;
  const int l_offset5 = -l_r + l_r * l_W;
  const int l_offset6 =    0 + l_r * l_W;
  const int l_offset7 = +l_r + l_r * l_W;

  const int l_offsetx = GRANULARITY / 2;
  const int l_offsety = GRANULARITY / 2;

  for (int l_y = l_r; l_y < l_H - l_r - 1; ++l_y)
  {
    const unsigned short * lp_line = lp_depth + (l_y * l_W + l_r);
    unsigned char * lp_norm = lp_normals + (l_y * l_W + l_r);

    for (int l_x = l_r; l_x < l_W - l_r - 1; ++l_x)
    {
      long l_d = lp_line[0];
      float *data = normal.ptr<float>(l_y * l_W + l_x);
      if (l_d < distance_threshold)
      {
        // accum
        long l_A[4]; l_A[0] = l_A[1] = l_A[2] = l_A[3] = 0;
        long l_b[2]; l_b[0] = l_b[1] = 0;
        accumBilateral(lp_line[l_offset0] - l_d, -l_r, -l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset1] - l_d,    0, -l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset2] - l_d, +l_r, -l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset3] - l_d, -l_r,    0, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset4] - l_d, +l_r,    0, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset5] - l_d, -l_r, +l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset6] - l_d,    0, +l_r, l_A, l_b, difference_threshold);
        accumBilateral(lp_line[l_offset7] - l_d, +l_r, +l_r, l_A, l_b, difference_threshold);

        // solve
        long l_det =  l_A[0] * l_A[3] - l_A[1] * l_A[1];
        long l_ddx =  l_A[3] * l_b[0] - l_A[1] * l_b[1];
        long l_ddy = -l_A[1] * l_b[0] + l_A[0] * l_b[1];

        /// @todo Magic number 1150 is focal length? This is something like
        /// f in SXGA mode, but in VGA is more like 530.
        float l_nx = static_cast<float>(cam[0] * l_ddx);
        float l_ny = static_cast<float>(cam[2] * l_ddy);
        float l_nz = static_cast<float>(-l_det * l_d);

        float l_sqrt = sqrtf(l_nx * l_nx + l_ny * l_ny + l_nz * l_nz);

        if (l_sqrt > 0)
        {
          float l_norminv = 1.0f / (l_sqrt);

          l_nx *= l_norminv;
          l_ny *= l_norminv;
          l_nz *= l_norminv;
            
          data[2] = (float)l_d;
            data[0] = (data[2] * (l_x - cam[1])) / cam[0];
            data[1] = (data[2] * (l_y - cam[3])) / cam[2];
          data[3] = l_nx;
          data[4] = l_ny;
          data[5] = l_nz;
          //*lp_norm = fabs(l_nz)*255;

          int l_val1 = static_cast<int>(l_nx * l_offsetx + l_offsetx);
          int l_val2 = static_cast<int>(l_ny * l_offsety + l_offsety);
          int l_val3 = static_cast<int>(l_nz * GRANULARITY + GRANULARITY);

          *lp_norm = NORMAL_LUT[l_val3][l_val2][l_val1];
        }
        // else
        // {
        //   *lp_norm = 0; // Discard shadows from depth sensor
        // }
      }
    //   else
    //   {
    //     *lp_norm = 0; //out of depth
    //   }
      ++lp_line;
      ++lp_norm;
    }
  }
  medianBlur(dst, dst, 5);
}

int main(int argc, char* argv[])
{
    Mat depth = cv::imread(argv[1], IMREAD_ANYDEPTH );
    // Mat img = cv::imread(argv[2]);
    int distance_threshold=1400;
    int difference_threshold = 50;
    float cam[4] = {572.4114, 325.2611, 573.57043, 242.04899};

    Mat quat_normal, pcn;
    quantizedNormals(depth, quat_normal, pcn, distance_threshold, difference_threshold, cam);

    writePLY(pcn, argv[2]);
    // cv::imshow("normal", quat_normal);
    // cv::waitKey(0);
    // cv::destroyAllWindows();
    return 0;
}