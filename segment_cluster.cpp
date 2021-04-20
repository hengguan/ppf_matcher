#include <iostream>
// #include "surface_matching.hpp"
// #include "surface_matching/ppf_helpers.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/filters/passthrough.h>

#include <vcg/complex/algorithms/pointcloud_normal.h>
#include <vcg/complex/complex.h>
// #include <wrap/io_trimesh/import_off.h>
// #include <wrap/io_trimesh/import.h>
// #include <wrap/io_trimesh/export_ply.h>
// #include <apps/QT/trimesh_QT_shared/mesh.h>

#include <thread>

#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>

using namespace std::chrono_literals;
using namespace std;
using namespace pcl;
using namespace vcg;

class MyEdge;
class MyFace;
class MyVertex;
struct MyUsedTypes : public UsedTypes<Use<MyVertex>::AsVertexType,
                                      Use<MyEdge>::AsEdgeType,
                                      Use<MyFace>::AsFaceType>
{
};

class MyVertex : public Vertex<MyUsedTypes, vertex::Coord3f, vertex::Normal3f, vertex::BitFlags>
{
};
class MyFace : public Face<MyUsedTypes, face::FFAdj, face::VertexRef, face::BitFlags>
{
};
class MyEdge : public Edge<MyUsedTypes>
{
};
class MyMesh : public tri::TriMesh<vector<MyVertex>, vector<MyFace>, vector<MyEdge>>
{
};

static void help(const string &errorMessage)
{
    cout << "Program init error : " << errorMessage << endl;
    cout << "\nUsage : ppf_normal_computation [input model file] [output model file]" << endl;
    cout << "\nPlease start again with new parameters" << endl;
}

// char filename[256][256];
// int len = 0;
int trave_dir(char* path, vector<string> &file_names, int depth)
{
    DIR *d; //声明一个句柄
    struct dirent *file; //readdir函数的返回值就存放在这个结构体中
    struct stat sb;   

    if(!(d = opendir(path)))
    {
        printf("error opendir %s!!!\n",path);
        return -1;
    }
    int idx=0;
    while((file = readdir(d)) != NULL)
    {
        //把当前目录.，上一级目录..及隐藏文件都去掉，避免死循环遍历目录
        if(strncmp(file->d_name, ".", 1) == 0)
            continue;
        // strcpy(file_names[idx++], file->d_name); //保存遍历到的文件名
        // file_names[idx++] = file->d_name;
        std::string str5 = file->d_name;
        file_names.push_back(str5);
        //判断该文件是否是目录，及是否已搜索了三层，这里我定义只搜索了三层目录，太深就不搜了，省得搜出太多文件
        // if(stat(file->d_name, &sb) >= 0 && S_ISDIR(sb.st_mode) && depth <= 1)
        // {
        //     trave_dir(file->d_name, depth + 1);
        // }
    }
    closedir(d);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        help("Not enough input arguments");
        exit(1);
    }

    // string modelFileName = (string)argv[1];
    // string outputFileName = (string)argv[2];
    // cv::Mat points, pointsAndNormals;
    vector<string> file_ids;
    trave_dir("../samples/data/cloud", file_ids, 1);

    boost::shared_ptr<visualization::PCLVisualizer> viewer(new visualization::PCLVisualizer("cloud_scene"));
    /*设置窗口viewer的背景颜色*/
    viewer->setBackgroundColor(0, 0, 0);

    for(int i=0; i<file_ids.size(); i++)
    {
        stringstream ss_in;
        ss_in << "../samples/data/cloud/" << file_ids[i];
        cout << "path: "<<ss_in.str()<<endl;

        PointCloud<PointXYZRGBA>::Ptr cloud(new PointCloud<PointXYZRGBA>);
        // PointCloud<PointXYZRGBA>::Ptr cloud_mm(new PointCloud<PointXYZRGBA>);
        PointCloud<PointXYZRGBA>::Ptr cloud_filtered(new PointCloud<PointXYZRGBA>);
        PointCloud<PointXYZ>::Ptr cloud_filtered_xyz(new PointCloud<PointXYZ>);
        PointCloud<PointXYZ>::Ptr cloud_sampled(new PointCloud<PointXYZ>);
        PointCloud<PointXYZ>::Ptr sor_cloud(new PointCloud<PointXYZ>);
        PointCloud<PointNormal>::Ptr cloud_cluster(new PointCloud<PointNormal>);
        // PointCloud<PointXYZ>::Ptr ext_cloud(new PointCloud<PointXYZ>);
        PointCloud<PointNormal>::Ptr cloud_point(new PointCloud<PointNormal>);
        PointCloud<PointNormal>::Ptr ext_cloud(new PointCloud<PointNormal>);
        PointCloud<PointNormal>::Ptr cloud_normal(new PointCloud<PointNormal>);

        if (pcl::io::loadPCDFile(ss_in.str(), *cloud) == -1)
        {
            PCL_ERROR("read false");
            return 0;
        }

        // Apply the filter
        pcl::FastBilateralFilter<pcl::PointXYZRGBA> fbf;
        fbf.setInputCloud (cloud);
        fbf.setSigmaS (5.0);
        fbf.setSigmaR (3.0);
        // fbf.setKeepOrganized(true);
        fbf.filter (*cloud_filtered);
        
        //3.直通滤波
        pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_(new pcl::PointCloud<pcl::PointXYZRGBA>);
        pcl::PassThrough<pcl::PointXYZRGBA> pass;     //创建滤波器对象
        pass.setInputCloud(cloud_filtered);                //设置待滤波的点云
        pass.setFilterFieldName("z");             //设置在Z轴方向上进行滤波
        pass.setFilterLimits(-1500, 0);    //设置滤波范围(从最高点向下12米去除)
        pass.setFilterLimitsNegative(false);      //保留
        pass.filter(*cloud_);

        copyPointCloud(*cloud_, *cloud_filtered_xyz);
        //体素化下采样******************************************************
        VoxelGrid<PointXYZ> vox;
        vox.setInputCloud(cloud_filtered_xyz);
        vox.setLeafSize(1.0, 1.0, 1.0);
        // vox.setKeepOrganized(true);
        vox.filter(*cloud_sampled);
        cout << "down sampling point cloud: " <<cloud_sampled->size()<< endl;

        //去除离群点********************************************************
        StatisticalOutlierRemoval<PointXYZ> sor;
        sor.setMeanK(50);
        sor.setInputCloud(cloud_sampled);
        sor.setStddevMulThresh(2.0);
        // sor.setKeepOrganized(true);
        sor.filter(*sor_cloud);
        cout << "Statistical Outlier Removal: " << sor_cloud->size()<< endl;

        // MyMesh m;
        // if (tri::io::ImporterOFF<MyMesh>::Open(m, argv[1]) != 0)
        // {
        //     printf("Error reading file  %s\n", argv[1]);
        //     exit(0);
        // }
        // vcg::tri::io::ImporterPLY<MyMesh>::Open(m, argv[1]);
        MyMesh m;
        m.Clear();

        int vertCount = sor_cloud->width * sor_cloud->height;
        vcg::tri::Allocator<MyMesh>::AddVertices(m, vertCount);
        for (int i = 0; i < vertCount; ++i)
            m.vert[i].P() = vcg::Point3f(sor_cloud->points[i].x, sor_cloud->points[i].y, sor_cloud->points[i].z);

        tri::PointCloudNormal<MyMesh>::Param p;
        p.fittingAdjNum = 10;
        p.smoothingIterNum = 1;
        p.viewPoint = {0.0, 0.0, 0.0};
        p.useViewPoint = true;
        tri::PointCloudNormal<MyMesh>::Compute(m, p, 0);
        
        for (int i = 0; i < m.vert.size(); ++i)
        {
            PointNormal np;
            vcg::Point3f n = m.vert[i].N();
            vcg::Point3f p = m.vert[i].P();
            np.x = p[0];
            np.y = p[1];
            np.z = p[2];
            np.normal_x = n[0];
            np.normal_y = n[1];
            np.normal_z = n[2];
            cloud_normal->push_back(np);
        }
        cout << "normal computation: " << cloud_normal->size() << endl;
        
        //欧式聚类*******************************************************
        // vector<PointIndices> ece_inlier;
        // search::KdTree<PointNormal>::Ptr tree(new search::KdTree<PointNormal>);
        // EuclideanClusterExtraction<PointNormal> ece;
        // ece.setInputCloud(cloud_normal);
        // ece.setClusterTolerance(80);
        // ece.setMinClusterSize(1000);
        // ece.setMaxClusterSize(200000);
        // ece.setSearchMethod(tree);
        // ece.extract(ece_inlier);
        // //聚类结果展示***************************************************
        ExtractIndices<PointNormal> ext;
        // int maxInd = 0, maxNum = 0;
        // cout << "number of clusters: " << ece_inlier.size() << endl;
        // for (int i = 0; i < ece_inlier.size(); i++)
        // {
        //     if (ece_inlier[i].indices.size() > maxNum)
        //     {
        //         maxInd = i;
        //         maxNum = ece_inlier[i].indices.size();
        //     }
        // }
        // vector<int> ece_inlier_ext = ece_inlier[maxInd].indices;
        // copyPointCloud(*cloud_normal, ece_inlier_ext, *cloud_cluster); //按照索引提取点云数据
        // cout << "Euclidean Cluster Extraction: " << cloud_cluster->size() << endl;

        //平面分割(RANSAC)********************************************************
        SACSegmentation<PointNormal> sac;
        PointIndices::Ptr inliner(new PointIndices);
        ModelCoefficients::Ptr coefficients(new ModelCoefficients);
        PointCloud<PointNormal>::Ptr sac_cloud(new PointCloud<PointNormal>);
        sac.setInputCloud(cloud_normal);
        sac.setMethodType(SAC_RANSAC);
        sac.setModelType(SACMODEL_PLANE);
        sac.setMaxIterations(100);
        sac.setDistanceThreshold(3);
        //提取平面(展示并输出)******************************************************
        ext.setInputCloud(cloud_normal);
        sac.segment(*inliner, *coefficients);
        if (inliner->indices.size() == 0)
        {
            std::cout << "segmentation failure!!" << endl;
        }
        //按照索引提取点云*************
        ext.setIndices(inliner);
        ext.setNegative(true);
        ext.filter(*ext_cloud);
        cout << "plane segmentation: " << ext_cloud->size() << endl;

        // save result model
        stringstream ss_out;
        ss_out << "../samples/data/cloud/" << file_ids[i] << "_normal.ply";
        pcl::io::savePLYFile(ss_out.str(), *ext_cloud);

        // viewer->addPointCloud<PointNormal>(ext_cloud, "cloud_scene");
        // viewer->setPointCloudRenderingProperties(visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_scene");

        // cout << "Loading points\n";
        // // cv::ppf_match_3d::loadPLYSimple(modelFileName.c_str(), 1).copyTo(points);
        // cv::Mat cloud_in = cv::Mat(sor_cloud->size(), 3, CV_32FC1);
        // for (int i = 0; i < sor_cloud->size(); i++)
        // {
        //     float *data = cloud_in.ptr<float>(i);
        //     data[0] = sor_cloud->at(i).x;
        //     data[1] = sor_cloud->at(i).y;
        //     data[2] = sor_cloud->at(i).z;
        // }

        // cout << "Computing normals\n";
        // cv::Vec3d viewpoint(0, 0, 0);
        // cv::ppf_match_3d::computeNormalsPC3d(cloud_in, pointsAndNormals, 10, false, viewpoint);

        // std::cout << "Writing points\n";
        // cv::ppf_match_3d::writePLY(pointsAndNormals, outputFileName.c_str());
        //the following function can also be used for debugging purposes
        //cv::ppf_match_3d::writePLYVisibleNormals(pointsAndNormals, outputFileName.c_str());

        std::cout << "Done\n";
        // while (!viewer->wasStopped())
        // {
        //     viewer->spinOnce(100);
        //     std::this_thread::sleep_for(100ms);
        // }
    }
    return 0;
}
