#include <iostream>
#include "surface_matching.hpp"
#include "surface_matching/ppf_helpers.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>

#include <thread>

using namespace std::chrono_literals;
using namespace std;
using namespace pcl;

static void help(const string &errorMessage)
{
    cout << "Program init error : " << errorMessage << endl;
    cout << "\nUsage : ppf_normal_computation [input model file] [output model file]" << endl;
    cout << "\nPlease start again with new parameters" << endl;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        help("Not enough input arguments");
        exit(1);
    }

    string modelFileName = (string)argv[1];
    string outputFileName = (string)argv[2];
    cv::Mat points, pointsAndNormals;

    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr cloud_sampled(new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr sor_cloud(new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr cloud_cluster(new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr ext_cloud(new PointCloud<PointXYZ>);

    if (io::loadPCDFile(argv[1], *cloud) == -1)
    {
        PCL_ERROR("read false");
        return 0;
    }

    //体素化下采样******************************************************
    VoxelGrid<PointXYZ> vox;
    vox.setInputCloud(cloud);
    vox.setLeafSize(2.0, 2.0, 2.0);
    vox.filter(*cloud_sampled);
    cout << "down sampling point cloud: " <<cloud_sampled->size()<< endl;
 
    //去除噪声点********************************************************
    StatisticalOutlierRemoval<PointXYZ> sor;
    sor.setMeanK(50);
    sor.setInputCloud(cloud_sampled);
    sor.setStddevMulThresh(3.0);
    sor.filter(*sor_cloud);
    cout << "Statistical Outlier Removal: " << sor_cloud->size()<< endl;

    //欧式聚类*******************************************************
    vector<PointIndices> ece_inlier;
    search::KdTree<PointXYZ>::Ptr tree(new search::KdTree<PointXYZ>);
    EuclideanClusterExtraction<PointXYZ> ece;
    ece.setInputCloud(sor_cloud);
    ece.setClusterTolerance(80);
    ece.setMinClusterSize(1000);
    ece.setMaxClusterSize(200000);
    ece.setSearchMethod(tree);
    ece.extract(ece_inlier);
    //聚类结果展示***************************************************
    ExtractIndices<PointXYZ> ext;
    int maxInd = 0, maxNum = 0;
    cout << "number of clusters: " << ece_inlier.size() << endl;
    for (int i = 0; i < ece_inlier.size(); i++)
    {
        if (ece_inlier[i].indices.size() > maxNum)
        {
            maxInd = i;
            maxNum = ece_inlier[i].indices.size();
        }
    }
    vector<int> ece_inlier_ext = ece_inlier[maxInd].indices;
    copyPointCloud(*sor_cloud, ece_inlier_ext, *cloud_cluster); //按照索引提取点云数据
    cout << "Euclidean Cluster Extraction: " << cloud_cluster->size()<< endl;

    //平面分割(RANSAC)********************************************************
    SACSegmentation<PointXYZ> sac;
    PointIndices::Ptr inliner(new PointIndices);
    ModelCoefficients::Ptr coefficients(new ModelCoefficients);
    PointCloud<PointXYZ>::Ptr sac_cloud(new PointCloud<PointXYZ>);
    sac.setInputCloud(cloud_cluster);
    sac.setMethodType(SAC_RANSAC);
    sac.setModelType(SACMODEL_PLANE);
    sac.setMaxIterations(100);
    sac.setDistanceThreshold(20);
    //提取平面(展示并输出)******************************************************
    ext.setInputCloud(cloud_cluster);
    sac.segment(*inliner, *coefficients);
    if (inliner->indices.size() == 0)
    {
        std::cout << "segmentation failure!!" << endl;
    }
    //按照索引提取点云*************
    ext.setIndices(inliner);
    ext.setNegative(true);
    ext.filter(*ext_cloud);
    cout << "plane segmentation: " << ext_cloud->size()<< endl;

    boost::shared_ptr<visualization::PCLVisualizer> viewer(new visualization::PCLVisualizer("cloud_scene"));
    /*设置窗口viewer的背景颜色*/
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<PointXYZ>(ext_cloud, "cloud_scene");
    viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_scene");

    cout << "Loading points\n";
    // cv::ppf_match_3d::loadPLYSimple(modelFileName.c_str(), 1).copyTo(points);
    cv::Mat cloud_in = cv::Mat(ext_cloud->size(), 3, CV_32FC1);
    for (int i = 0; i < ext_cloud->size(); i++)
    {
        float *data = cloud_in.ptr<float>(i);
        data[0] = ext_cloud->at(i).x;
        data[1] = ext_cloud->at(i).y;
        data[2] = ext_cloud->at(i).z;
    }

    cout << "Computing normals\n";
    cv::Vec3d viewpoint(0, 0, 0);
    cv::ppf_match_3d::computeNormalsPC3d(cloud_in, pointsAndNormals, 6, false, viewpoint);

    std::cout << "Writing points\n";
    cv::ppf_match_3d::writePLY(pointsAndNormals, outputFileName.c_str());
    //the following function can also be used for debugging purposes
    //cv::ppf_match_3d::writePLYVisibleNormals(pointsAndNormals, outputFileName.c_str());

    std::cout << "Done\n";
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(100ms);
    }
    return 0;
}
