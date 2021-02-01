#include "surface_matching.hpp"
#include <iostream>
#include "surface_matching/ppf_helpers.hpp"
#include "opencv2/core/utility.hpp"

#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>

// #include <boost/thread.hpp>
// #include <boost/date_time.hpp>
#include <thread>

// #include <pcl/features/normal_3d.h>
// #include <pcl/console/parse.h>

using namespace std::chrono_literals;
using namespace std;
using namespace cv;
using namespace ppf_match_3d;
using namespace pcl;
using namespace boost;

static void help(const string &errorMessage)
{
    cout << "Program init error : " << errorMessage << endl;
    cout << "\nUsage : ppf_matching [input model file] [input scene file]" << endl;
    cout << "\nPlease start again with new parameters" << endl;
}

int main(int argc, char **argv)
{
    // welcome message
    cout << "****************************************************" << endl;
    cout << "* Surface Matching demonstration : demonstrates the use of surface matching"
            " using point pair features."
         << endl;
    cout << "* The sample loads a model and a scene, where the model lies in a different"
            " pose than the training.\n* It then trains the model and searches for it in the"
            " input scene. The detected poses are further refined by ICP\n* and printed to the "
            " standard output."
         << endl;
    cout << "****************************************************" << endl;

    if (argc < 3)
    {
        help("Not enough input arguments");
        exit(1);
    }

#if (defined __x86_64__ || defined _M_X64)
    cout << "Running on 64 bits" << endl;
#else
    cout << "Running on 32 bits" << endl;
#endif

#ifdef _OPENMP
    cout << "Running with OpenMP" << endl;
#else
    cout << "Running without OpenMP and without TBB" << endl;
#endif

    string modelFileName = (string)argv[1];
    string sceneFileName = (string)argv[2];

    boost::shared_ptr<visualization::PCLVisualizer> viewer(new visualization::PCLVisualizer("3D viewer"));
    /*设置窗口viewer的背景颜色*/
    viewer->setBackgroundColor(0, 0, 0);

    Mat pc = loadPLYSimple(modelFileName.c_str(), 1);
    PointCloud<PointNormal>::Ptr cloud_normal(new PointCloud<PointNormal>());
    PointCloud<PointNormal>::Ptr vox_cloud(new PointCloud<PointNormal>());
    for (int i = 0; i < pc.rows; i++)
    {
        const float *data = pc.ptr<float>(i);
        PointNormal pn;
        pn.x = data[0];
        pn.y = data[1];
        pn.z = data[2];
        pn.normal_x = data[3];
        pn.normal_y = data[4];
        pn.normal_z = data[5];
        cloud_normal->push_back(pn);
    }

    VoxelGrid<PointNormal> vox;
    vox.setInputCloud(cloud_normal);
    vox.setLeafSize(6.0, 6.0, 6.0);
    vox.filter(*vox_cloud);

    Mat pc_in = Mat(vox_cloud->size(), 6, CV_32FC1);
    for(int i=0;i<vox_cloud->size();i++)
    {
        float *data = pc_in.ptr<float>(i);
        data[0] = vox_cloud->at(i).x;
        data[1] = vox_cloud->at(i).y;
        data[2] = vox_cloud->at(i).z;
        data[3] = vox_cloud->at(i).normal_x;
        data[4] = vox_cloud->at(i).normal_y;
        data[5] = vox_cloud->at(i).normal_z;
    }
    std::cout << "number of model points: " << pc_in.rows << std::endl;

    // Now train the model
    cout << "Training..." << endl;
    int64 tick1 = cv::getTickCount();
    ppf_match_3d::PPF3DDetector detector(0.025, 0.05);
    detector.trainModel(pc_in);
    int64 tick2 = cv::getTickCount();
    cout << endl
         << "Training complete in "
         << (double)(tick2 - tick1) / cv::getTickFrequency()
         << " sec" << endl
         << "Loading model..." << endl;

    // Read the scene
    Mat pcTest = loadPLYSimple(sceneFileName.c_str(), 1);
    std::cout << "number of scene points: " << pc.rows << std::endl;
    PointCloud<PointXYZ>::Ptr cloud_scene(new PointCloud<PointXYZ>());
    for (int i = 0; i < pcTest.rows; i++)
    {
        const float *data = pcTest.ptr<float>(i);
        PointXYZ pt;
        pt.x = data[0];
        pt.y = data[1];
        pt.z = data[2];
        cloud_scene->push_back(pt);
    }
    viewer->addPointCloud<PointXYZ>(cloud_scene, "cloud_scene");
    viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_scene");

    // Match the model to the scene and get the pose
    cout << endl
         << "Starting matching..." << endl;
    vector<Pose3DPtr> results;
    tick1 = cv::getTickCount();
    detector.match(pcTest, results, 1.0 / 10.0, 0.025);
    tick2 = cv::getTickCount();
    cout << endl
         << "PPF Elapsed Time " << (tick2 - tick1) / cv::getTickFrequency() << " sec" << endl;

    //check results size from match call above
    size_t results_size = results.size();
    cout << "Number of matching poses: " << results_size;
    if (results_size == 0)
    {
        cout << endl
             << "No matching poses found. Exiting." << endl;
        exit(0);
    }

    // Get only first N results - but adjust to results size if num of results are less than that specified by N
    size_t N = 2;
    if (results_size < N)
    {
        cout << endl
             << "Reducing matching poses to be reported (as specified in code): "
             << N << " to the number of matches found: " << results_size << endl;
        N = results_size;
    }
    vector<Pose3DPtr> resultsSub(results.begin(), results.begin() + N);

    // Create an instance of ICP
    ICP icp(100, 0.005f, 2.5f, 8);
    int64 t1 = cv::getTickCount();

    // Register for all selected poses
    cout << endl
         << "Performing ICP on " << N << " poses..." << endl;
    icp.registerModelToScene(pc, pcTest, resultsSub);
    int64 t2 = cv::getTickCount();

    cout << endl
         << "ICP Elapsed Time " << (t2 - t1) / cv::getTickFrequency() << " sec" << endl;

    cout << "Poses: " << endl;
    // debug first five poses
    for (size_t i = 0; i < resultsSub.size(); i++)
    {
        Pose3DPtr result = resultsSub[i];
        cout << "Pose Result " << i << endl;
        result->printPose();
        if (i >= 0)
        {
            Mat pct = transformPCPose(pc, result->pose);
            // writePLY(pct, "para6700PCTrans.ply");
            PointCloud<PointXYZ>::Ptr cloud_result(new PointCloud<PointXYZ>());
            for (int i = 0; i < pct.rows; i++)
            {
                const float *data = pct.ptr<float>(i);
                PointXYZ pt;
                pt.x = data[0];
                pt.y = data[1];
                pt.z = data[2];
                cloud_result->push_back(pt);
            }
            stringstream ss;
            ss << "pose_" << i;
            pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> RandomColor(cloud_result);
            viewer->addPointCloud<PointXYZ>(cloud_result, RandomColor, ss.str());
            /*修改现实点云的尺寸。用户可通过该方法控制点云在视窗中的显示方式*/
            viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 1, ss.str());
        }
    }

    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(100ms);
    }
    return 0;
}
