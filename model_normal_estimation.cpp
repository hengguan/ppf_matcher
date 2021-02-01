#include <iostream>
#include "surface_matching.hpp"
#include "surface_matching/ppf_helpers.hpp"

using namespace std;

static void help(const string& errorMessage)
{
  cout << "Program init error : " << errorMessage << endl;
  cout << "\nUsage : ppf_normal_computation [input model file] [output model file]" << endl;
  cout << "\nPlease start again with new parameters" << endl;
}

int main(int argc, char** argv)
{
  if (argc < 3)
  {
    help("Not enough input arguments");
    exit(1);
  }

  string modelFileName = (string)argv[1];
  string outputFileName = (string)argv[2];
  cv::Mat points, pointsAndNormals;

  cout << "Loading points\n";
  cv::ppf_match_3d::loadPLYSimple(modelFileName.c_str(), 1).copyTo(points);

  cout << "Computing normals\n";
  cv::Vec3d viewpoint(0, 0, 0);
  cv::ppf_match_3d::computeNormalsPC3d(points, pointsAndNormals, 6, false, viewpoint);

  std::cout << "Writing points\n";
  cv::ppf_match_3d::writePLY(pointsAndNormals, outputFileName.c_str());
  //the following function can also be used for debugging purposes
  //cv::ppf_match_3d::writePLYVisibleNormals(pointsAndNormals, outputFileName.c_str());

  std::cout << "Done\n";
  return 0;
}
