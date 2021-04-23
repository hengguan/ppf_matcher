#include "surface_matching.hpp"
#include <iostream>
#include "surface_matching/ppf_helpers.hpp"
#include "opencv2/core/utility.hpp"

#include <pcl/common/common_headers.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>

// #include <boost/thread.hpp>
// #include <boost/date_time.hpp>
#include <thread>

// #include <pcl/features/normal_3d.h>
// #include <pcl/console/parse.h>
#include <yaml-cpp/yaml.h>

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

    YAML::Node cfg = YAML::LoadFile("../config.yaml");
    double voxSize = cfg["voxSize"].as<double>();
    double relativeSamStep = cfg["relativeSamStep"].as<double>();
    double relativeDistStep = cfg["relativeDistStep"].as<double>();
    double sceneSamStep = cfg["sceneSamStep"].as<double>();
    double sceneDistStep = cfg["sceneDistStep"].as<double>();

    // string modelFileName = (string)argv[1];
    // string sceneFileName = (string)argv[2];

    // Mat pc = loadPLYSimple(modelFileName.c_str(), 1);
    PointCloud<PointNormal>::Ptr cloud_normal(new PointCloud<PointNormal>());
    PointCloud<PointNormal>::Ptr vox_cloud(new PointCloud<PointNormal>());
    // for (int i = 0; i < pc.rows; i++)
    // {
    //     const float *data = pc.ptr<float>(i);
    //     PointNormal pn;
    //     pn.x = data[0];
    //     pn.y = data[1];
    //     pn.z = data[2];
    //     pn.normal_x = data[3];
    //     pn.normal_y = data[4];
    //     pn.normal_z = data[5];
    //     cloud_normal->push_back(pn);
    // }
    if (pcl::io::loadPLYFile(argv[1], *cloud_normal) == -1)
    {
        PCL_ERROR("read false");
        return 0;
    }

    // VoxelGrid<PointNormal> vox;
    // vox.setInputCloud(cloud_normal);
    // vox.setLeafSize(voxSize, voxSize, voxSize);
    // vox.filter(*vox_cloud);

    Mat pc_in = Mat(cloud_normal->size(), 6, CV_32FC1);
    for(int i=0;i<cloud_normal->size();i++)
    {
        float *data = pc_in.ptr<float>(i);
        data[0] = cloud_normal->at(i).x*1000.0;
        data[1] = cloud_normal->at(i).y*1000.0;
        data[2] = cloud_normal->at(i).z*1000.0;
        data[3] = cloud_normal->at(i).normal_x;
        data[4] = cloud_normal->at(i).normal_y;
        data[5] = cloud_normal->at(i).normal_z;
    }
    std::cout << "number of model points: " << pc_in.rows << std::endl;

    // Now train the model
    cout << "Training..." << endl;
    int64 tick1 = cv::getTickCount();
    ppf_match_3d::PPF3DDetector detector(relativeSamStep, relativeDistStep);
    detector.trainModel(pc_in);
    int64 tick2 = cv::getTickCount();
    cout << endl
        << "Training complete in "
        << (double)(tick2 - tick1) / cv::getTickFrequency()
        << " sec" << endl
        << "Loading model..." << endl;
    
    // vector<string> file_ids = {
    //     "000041.pcd_normal", "000129.pcd_normal", "000310.pcd_normal", 
    //     "000368.pcd_normal", "000382.pcd_normal", "000476.pcd_normal", 
    //     "000480.pcd_normal", "000528.pcd_normal", "000671.pcd_normal", "000694.pcd_normal"};
    // vector<string> file_ids = {
    //     "000069.pcd_normal", "000082.pcd_normal", "000170.pcd_normal", 
    //     "000300.pcd_normal", "000377.pcd_normal", "000499.pcd_normal", 
    //     "000684.pcd_normal", "000721.pcd_normal", "000739.pcd_normal", "000745.pcd_normal"};
    vector<string> file_ids = {"000041m", "000145m","000283m", "000403m", "000512m", "000651m", "000753m", "000902m", "000992m", "001080m", "001168m", "001231m"};
    // vector<string> file_ids = {"scene_0", "scene_1","scene_2", "scene_3", "scene_4", "scene_5", "scene_6", "scene_7", "scene_8", "scene_9", "scene_10", "scene_11"};
    // vector<string> file_ids = {"000041m", "000044m","000063m", "000098m", "000123m", "000142m", "000177m", "000181m", "000184m", "000280m"};

    // Create an instance of ICP
    ICP icp(100, 0.05f, 2.5f, 8);
    for(int i=0; i<file_ids.size(); i++)
    {
        boost::shared_ptr<visualization::PCLVisualizer> viewer(new visualization::PCLVisualizer("3D viewer"));
        /*设置窗口viewer的背景颜色*/
        viewer->setBackgroundColor(0, 0, 0);
        stringstream ss_in;
        ss_in << "../samples/data/cloud_lm/" << file_ids[i] << ".ply";
        // Read the scene
        // Mat pcTest = loadPLYSimple(sceneFileName.c_str(), 1);
        PointCloud<PointNormal>::Ptr scene_normal(new PointCloud<PointNormal>());
        if (pcl::io::loadPLYFile(ss_in.str(), *scene_normal) == -1)
        {
            PCL_ERROR("read false");
            return 0;
        }
        Mat pcTest = Mat(scene_normal->size(), 6, CV_32FC1);
        for(int i=0;i<scene_normal->size();i++)
        {
            float *data = pcTest.ptr<float>(i);
            scene_normal->at(i).x *= 1000.0;
            scene_normal->at(i).y *= 1000.0;
            scene_normal->at(i).z *= 1000.0;
            data[0] = scene_normal->at(i).x;
            data[1] = scene_normal->at(i).y;
            data[2] = scene_normal->at(i).z;
            data[3] = scene_normal->at(i).normal_x;
            data[4] = scene_normal->at(i).normal_y;
            data[5] = scene_normal->at(i).normal_z;
        }
        std::cout << "number of scene points: " << pcTest.rows << std::endl;
        // PointCloud<PointXYZ>::Ptr cloud_scene(new PointCloud<PointXYZ>());
        // for (int i = 0; i < pcTest.rows; i++)
        // {
        //     const float *data = pcTest.ptr<float>(i);
        //     PointXYZ pt;
        //     pt.x = data[0];
        //     pt.y = data[1];
        //     pt.z = data[2];
        //     cloud_scene->push_back(pt);
        // }

        viewer->removeAllShapes();
		viewer->removeAllPointClouds();
        stringstream ssc;
        ssc << "cloud_scene_" << i;
        // viewer->removePointCloud(ssc.str());
        viewer->addPointCloud<PointNormal>(scene_normal, ssc.str());
        viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 1, ssc.str());

        // Match the model to the scene and get the pose
        cout << endl
            << "Starting matching..." << endl;
        vector<Pose3DPtr> results;
        tick1 = cv::getTickCount();
        detector.match(pcTest, results, sceneSamStep, sceneDistStep);
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
        size_t N = 1;
        if (results_size < N)
        {
            cout << endl
                << "Reducing matching poses to be reported (as specified in code): "
                << N << " to the number of matches found: " << results_size << endl;
            N = results_size;
        }
        vector<Pose3DPtr> resultsSub(results.begin(), results.begin() + N);

        int64 t1 = cv::getTickCount();
        // Register for all selected poses
        cout << endl
            << "Performing ICP on " << N << " poses..." << endl;
        icp.registerModelToScene(detector.sampled_refine, pcTest, resultsSub);
        int64 t2 = cv::getTickCount();

        cout << endl
            << "ICP Elapsed Time " << (t2 - t1) / cv::getTickFrequency() << " sec" << endl;

        cout << "Poses: " << endl;
        vector<vector<int>> color_map = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {0, 255, 255}, {255, 0, 255}};
        int color_idx = 0;
        // debug first five poses
        for (size_t j = 0; j < resultsSub.size(); j++)
        {
            Pose3DPtr result = resultsSub[j];
            if (result->residual > 9.0)
                continue;
            vector<int> color = color_map.at(color_idx);
            color_idx ++;
            result->printPose();
            // if (j >= 0)
            // {
                Mat pct = transformPCPose(pc_in, result->pose);
                // writePLY(pct, "para6700PCTrans.ply");
                PointCloud<PointXYZ>::Ptr cloud_result(new PointCloud<PointXYZ>());
                for (int k = 0; k < pct.rows; k++)
                {
                    const float *data = pct.ptr<float>(k);
                    PointXYZ pt;
                    pt.x = data[0];
                    pt.y = data[1];
                    pt.z = data[2];
                    cloud_result->push_back(pt);
                }
                stringstream ss;
                ss << "pose_" << i*10+j;
                // pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ> RandomColor(cloud_result);
                pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud_result, color.at(0), color.at(1), color.at(2)); // green
                viewer->addPointCloud<PointXYZ>(cloud_result, single_color, ss.str());
                /*修改现实点云的尺寸。用户可通过该方法控制点云在视窗中的显示方式*/
                viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 1, ss.str());
            // }
            // break;
        }

        while (!viewer->wasStopped())
        {
            viewer->spinOnce(100);
            std::this_thread::sleep_for(100ms);
        }
        // getchar();
    }
    return 0;
}
