// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
// kdtree_incremental_time为kdtree建立时间，kdtree_search_time为kdtree搜索时间，kdtree_delete_time为kdtree删除时间;
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
// T1为雷达初始时间戳，s_plot为整个流程耗时，s_plot2特征点数量,s_plot3为kdtree增量时间，s_plot4为kdtree搜索耗时，s_plot5为kdtree删除点数量
//，s_plot6为kdtree删除耗时，s_plot7为kdtree初始大小，s_plot8为kdtree结束大小,s_plot9为平均消耗时间，s_plot10为添加点数量，s_plot11为点云预处理的总时间
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
// 定义全局变量，用于记录时间,match_time为匹配时间，solve_time为求解时间，solve_const_H_time为求解H矩阵时间
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
// kdtree_size_st为ikd-tree获得的节点数，kdtree_size_end为ikd-tree结束时的节点数，add_point_size为添加点的数量，kdtree_delete_counter为删除点的数量
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
// runtime_pos_log运行时的log是否开启，pcd_save_en是否保存pcd文件，time_sync_en是否同步时间
bool   runtime_pos_log = false, pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0}; //残差，点到面距离平方和
float DET_RANGE = 300.0f; //设置的当前雷达系中心到各个地图边缘的距离
const float MOV_THRESHOLD = 1.5f; //设置的当前雷达系中心到各个地图边缘的权重
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer; // 互斥锁
condition_variable sig_buffer; // 条件变量

string root_dir = ROOT_DIR; //设置根目录
string map_file_path, lid_topic, imu_topic; //设置地图文件路径，雷达topic，imu topic

double res_mean_last = 0.05, total_residual = 0.0; //设置残差平均值，残差总和
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0; //设置雷达时间戳，imu时间戳
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001; //设置imu的角速度协方差，加速度协方差，角速度偏置协方差，加速度偏置协方差
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0; //设置滤波器的最小尺寸，地图的最小尺寸，视野角度
 //设置立方体长度，视野一半的角度，视野总角度，总距离，雷达结束时间，第一帧雷达时间戳
double cube_len = 0, HALF_FOV_COS = 0, FOV_DEG = 0, total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
//设置有效特征点数，时间log计数器, scan_count：接收到的激光雷达Msg的总数，publish_count：接收到的IMU的Msg的总数
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0; 
//设置迭代次数，下采样的点数，最大迭代次数，有效点数
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_save_interval = -1, pcd_index = 0;
bool   point_selected_surf[100000] = {0}; // 是否为平面特征点
// lidar_pushed：用于判断激光雷达数据是否从缓存队列中拿到meas中的数据, flg_EKF_inited用于判断EKF是否初始化完成
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
//设置是否发布激光雷达数据，是否发布稠密数据，是否发布激光雷达数据在body系下的数据
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

vector<vector<int>>  pointSearchInd_surf;   //每个点的索引,暂时没用到
vector<BoxPointType> cub_needrm; // ikd-tree中，地图需要移除的包围盒序列
vector<PointVector>  Nearest_Points; //每个点的最近点序列
vector<double>       extrinT(3, 0.0); //雷达相对于IMU的外参t
vector<double>       extrinR(9, 0.0); //雷达相对于IMU的外参R
deque<double>                     time_buffer; // 激光雷达数据时间戳缓存队列
deque<PointCloudXYZI::Ptr>        lidar_buffer; //记录特征提取或间隔采样后的lidar（特征）数据
deque<sensor_msgs::Imu::ConstPtr> imu_buffer; // IMU数据缓存队列

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI()); //提取地图中的特征点，IKD-tree获得
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI()); //去畸变后的点云特征
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI()); //畸变纠正后降采样的单帧点云，lidar系
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI()); //畸变纠正后降采样的单帧点云，w系
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1)); //特征点在地图中对应点的，局部平面参数,w系
PointCloudXYZI::Ptr laserCloudOri(new PointCloudXYZI(100000, 1)); // laserCloudOri是畸变纠正后降采样的单帧点云，body系
PointCloudXYZI::Ptr corr_normvect(new PointCloudXYZI(100000, 1)); //对应点法相量
PointCloudXYZI::Ptr _featsArray; // ikd-tree中，map需要移除的点云序列

pcl::VoxelGrid<PointType> downSizeFilterSurf; //单帧内降采样使用voxel grid
pcl::VoxelGrid<PointType> downSizeFilterMap; //未使用

KD_TREE<PointType> ikdtree; // ikd-tree类

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0); //雷达相对于body系的X轴方向的点
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0); //雷达相对于world系的X轴方向的点
V3D euler_cur; //当前的欧拉角
V3D position_last(Zero3d); //上一帧的位置
V3D Lidar_T_wrt_IMU(Zero3d); // T lidar to imu (imu = r * lidar + t)
M3D Lidar_R_wrt_IMU(Eye3d); // R lidar to imu (imu = r * lidar + t)

/*** EKF inputs and output ***/
MeasureGroup Measures; // 当前帧预处理后的点云，以及当前帧内所有的IMU数据
esekfom::esekf<state_ikfom, 12, input_ikfom> kf; // 系统状态，噪声维度(12维)，输入(加速度和角速度测量值)
state_ikfom state_point; // 系统状态(点)
vect3 pos_lid; // 在world系下的Lidar坐标

// //输出的路径参数
nav_msgs::Path path; //包含了一系列位姿
nav_msgs::Odometry odomAftMapped; //只包含了一个位姿
geometry_msgs::Quaternion geoQuat; //四元数
geometry_msgs::PoseStamped msg_body_pose; //位姿

shared_ptr<Preprocess> p_pre(new Preprocess()); // 定义指向激光雷达数据的预处理类Preprocess的智能指针
shared_ptr<ImuProcess> p_imu(new ImuProcess()); // 定义指向IMU数据预处理类ImuProcess的智能指针

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp)  
{
    V3D rot_ang(Log(state_point.rot.toRotationMatrix()));
    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    fprintf(fp, "%lf %lf %lf ", state_point.pos(0), state_point.pos(1), state_point.pos(2)); // Pos  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
    fprintf(fp, "%lf %lf %lf ", state_point.vel(0), state_point.vel(1), state_point.vel(2)); // Vel  
    fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
    fprintf(fp, "%lf %lf %lf ", state_point.bg(0), state_point.bg(1), state_point.bg(2));    // Bias_g  
    fprintf(fp, "%lf %lf %lf ", state_point.ba(0), state_point.ba(1), state_point.ba(2));    // Bias_a  
    fprintf(fp, "%lf %lf %lf ", state_point.grav[0], state_point.grav[1], state_point.grav[2]); // Bias_a  
    fprintf(fp, "\r\n");  
    fflush(fp);
}

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

// 先从Lidar系到IMU系下，再从IMU系下转到World系下
void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

/// @brief 将pi转到当前帧IMU系相对于世界系的坐标po
/// @tparam T 
/// @param pi 
/// @param po 
template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

void points_cache_collect()
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    // for (int i = 0; i < points_history.size(); i++) _featsArray->push_back(points_history[i]);
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;
/// @brief 在ESKF前向传播的状态之上，动态调整地图区域，防止地图过大导致内存爆炸
/// 和LOAM中进行局部地图的提取方法类似
void lasermap_fov_segment()
{
    cub_needrm.clear(); // ikd-tree中，地图需要移除的包围盒序列，完成上次移除后进行清空
    kdtree_delete_counter = 0; // 记录kdtree 需要删除的次数
    kdtree_delete_time = 0.0;    // 记录kdtree 需要删除的时间
    // 将XAxisPoint_body转到当前帧IMU系相对世界系的坐标XAxisPoint_world
    pointBodyToWorld(XAxisPoint_body, XAxisPoint_world);
    V3D pos_LiD = pos_lid; // 当前帧相对于World系下的LiDAR坐标
    // 局部地图未初始化，需要进行初始化，完成初始化后直接返回
    // 初始化局部地图包围盒角点，以w系下lidar位置为中心，得到长宽高200*200*200的局部地图
    if (!Localmap_Initialized)
    {
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0; // cube_len = 200 默认值
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    // 完成初始化后，
    float dist_to_map_edge[3][2]; // 记录各个方向上中心和局部地图的边界，形象理解就是中心距离立方体盒子六个面的距离
    bool need_move = false; // 记录局部地图中心点是否需要移动
    // 计算当前帧LiDAR系到局部地图边缘的距离
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        // MOV_THRESHOLD = 1.5 设置的当前雷达系中心到各个地图边缘的权重
        // DET_RANGE = 300.0f; 设置的当前雷达系中心到各个地图边缘的距离
        // 如果某个方向上的边界距离超过了阈值大小，判断是否需要移动
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
    }
    if (!need_move) return; // 未超出边界，不需要移动，直接返回
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points; // 记录新盒子的边界点
    // 计算局部地图中心需要移动的距离
    // TODO：？？？ 为什么cube_len = 200，但是DET_RANGE = 300，而且权重MOV_THRESHOLD = 1.5,这样的话
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints); // 统计需要删除BOX中的序列范围
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    double delete_begin = omp_get_wtime();
    // 根据统计的删除范围，从kdtree中进行点云的删除
    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
    kdtree_delete_time = omp_get_wtime() - delete_begin;
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0; // 用于记录LiDAR和IMU的时间差
bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    mtx_buffer.lock(); // 进行雷达数据提取和处理时，将相关buffer锁住，这样就不会有新进入的信息干扰
    double preprocess_start_time = omp_get_wtime(); // 记录当前的时间(单位：秒)——记录点云预处理耗费时间
    scan_count ++; // 雷达处理数量 + 1——用于记录激光雷达扫描的总次数
    // Step 1 当前帧雷达的时间戳 小于 上一帧雷达数据时间戳，说明数据有问题，直接清空lidar_buffer
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec(); // 记录当前帧雷达数据时间戳
    
    // Step 2 IMU和雷达时间戳对齐
    // time_sync_en = true ：进行时间戳对齐(但是在FAST-LIO中，一般是false)
    // abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 这里的last_timestamp_lidar实际上就是当前帧雷达时间戳：在不需要进行时间同步的时候，要求IMU和激光雷达数据的时间差不能大于10s，否则输出IMU和LiDAR时间不对齐的错误信息
    // imu lidar都不为空
    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    // 如果执行时间同步
    // IMU和雷达时间戳差值大于1s，便标记已经进行过时间同步了（仅计算一次(!timediff_set_flg=false，就不会再进入这个if了)）
    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    // Step 3 雷达数据预处理
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr); // 将预处理完之后的雷达数据push进lidar_buffer
    time_buffer.push_back(last_timestamp_lidar); // 记录雷达数据的时间戳
    
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

/// @brief IMU回调函数，接收IMU数据，对齐IMU和Lidar的时间戳，并且删除非法数据，将合法的IMU保存在IMU Buffer中
/// @param msg_in 
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count ++; // 回调IMU总次数记录
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    // 如果需要将IMU和激光数据的时间戳对齐
    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        // TODO：直接将IMU数据对齐到激光雷达数据？？？这样不会改变积分结构吗？或者后续有额外的处理？？？
        msg->header.stamp = \
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    // 默认情况下time_offset_lidar_to_imu = 0.0 —— 这是先验信息，提前知道雷达和IMU时间戳的差异
    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);

    double timestamp = msg->header.stamp.toSec(); // 当前IMU数据时间戳信息

    mtx_buffer.lock(); // 锁住当前buffer

    // 如果上一个IMU时间戳 大于 当前IMU时间戳，说明IMU数据出现问题，Buff清空
    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
    }

    last_timestamp_imu = timestamp; // 更新IMU上一帧数据时间戳

    imu_buffer.push_back(msg); // 将当前IMU数据存入到缓冲器中
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
/// @brief 进行激光雷达数据和IMU数据之间的时间戳对齐
/// 函数处理imu和lidar接收buffer中的数据。主要是将当前帧激光点云，以及这当前帧点云扫描期间的imu数据完成时间戳对齐后 一起打包放到MeasureGroup里。 
/// @param[in out] meas 从相关buffer中取出的数据都存入meas中
/// @return 是否成功对齐时间戳
bool sync_packages(MeasureGroup &meas)
{
    // Step 1 如果激光和IMU其中由一个为空，那么直接退出
    if (lidar_buffer.empty() || imu_buffer.empty()) 
    {
        return false;
    }

    /*** push a lidar scan ***/
    // Step 2 将预处理之后的雷达点云数据push进meas中
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front(); // 取出最开始的雷达数据
        meas.lidar_beg_time = time_buffer.front(); // 雷达数据开始时间戳
        // 如果点云数量过少
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        // meas.lidar->points.back().curvature / double(1000) = 点云结束时间（除以1000，毫秒化？） 小于 当前点云平均扫描时间
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            // TODO
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        // 当前雷达数据结束时间戳
        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }
    
    // Step 3 当前IMU数据时间戳 不能 小于 该次雷达扫描结束时间，否则无法进行后续的反向传播
    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    // Step 4 将可用的IMU数据存储到meas中：将所有lidar_beg_time和lidar_end_time之间的IMU数据取出
    // TODO：论文里说：不是雷达频率比IMU快吗？
    // 答：激光雷达一次扫描的时间肯定比IMU返回一次的时间长，关于论文里说得点云频率远远快于IMU，指得是：
    //          扫描一周中，即SCAN_RATE的频率远远快于IMU的频率
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear(); // 清空旧数据
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    // 取出后，就从buffer中清理掉
    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        /* transform to world frame */
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
        /* decide if need add to map */
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point; 
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
    kdtree_incremental_time = omp_get_wtime() - st_time;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en)
    {
        int size = feats_undistort->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            RGBpointBodyToWorld(&feats_undistort->points[i], \
                                &laserCloudWorld->points[i]);
        }
        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num ++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0  && scan_wait_num >= pcd_save_interval)
        {
            pcd_index ++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_effect_world(const ros::Publisher & pubLaserCloudEffect)
{
    PointCloudXYZI::Ptr laserCloudWorld( \
                    new PointCloudXYZI(effct_feat_num, 1));
    for (int i = 0; i < effct_feat_num; i++)
    {
        RGBpointBodyToWorld(&laserCloudOri->points[i], \
                            &laserCloudWorld->points[i]);
    }
    sensor_msgs::PointCloud2 laserCloudFullRes3;
    pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
    laserCloudFullRes3.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudFullRes3.header.frame_id = "camera_init";
    pubLaserCloudEffect.publish(laserCloudFullRes3);
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;
    
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);// ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

/// @brief 计算点云匹配的残差（即观测h）以及相对于x的雅克比矩阵（即论文中的H） 
/// @param s 当前系统状态
/// @param ekfom_data 包含观测值以及H，和H相对于x/v的相关导数模型
void h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    double match_start = omp_get_wtime(); // 当前系统时间
    laserCloudOri->clear(); 
    corr_normvect->clear(); 
    total_residual = 0.0; 

    /** closest surface search and residual computation **/
    // 最邻近点搜索和残差计算(点到面的距离)
    #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
    #endif
    for (int i = 0; i < feats_down_size; i++) // feats_down_size ：降采样后的点云数量
    {
        PointType &point_body  = feats_down_body->points[i]; // 降采样后点在LiDAR系下的坐标
        PointType &point_world = feats_down_world->points[i]; // 存储降采样后点在World系下的坐标

        /* transform to world frame */
        // 将点云从LiDAR系转到World系下
        V3D p_body(point_body.x, point_body.y, point_body.z);
        V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);
        point_world.x = p_global(0);
        point_world.y = p_global(1);
        point_world.z = p_global(2);
        point_world.intensity = point_body.intensity;

        vector<float> pointSearchSqDis(NUM_MATCH_POINTS); // 存储最邻近的5个点距离point_world的距离

        // 之前对Nearest_Points已经进行了内存分配了，所以这里直接引用索引即可
        auto &points_near = Nearest_Points[i]; // 存储点的邻近点

        // 状态迭代更新是否收敛
        if (ekfom_data.converge)
        {
            /** Find the closest surfaces in the map **/
            // 在局部地图中找到point_world最邻近的点
            ikdtree.Nearest_Search(point_world, NUM_MATCH_POINTS, points_near, pointSearchSqDis);
            // 如果搜索到的邻近点的数量小于5 或者 邻近点距离point_world最大距离超过5米，认为该点不是有效点
            // 到这里还不能判断是否是平面点，只是判断该点是否有效
            point_selected_surf[i] = points_near.size() < NUM_MATCH_POINTS ? false : pointSearchSqDis[NUM_MATCH_POINTS - 1] > 5 ? false : true;
        }

        // 如果不是有效点，直接跳过该点
        if (!point_selected_surf[i]) continue;

        VF(4) pabcd; // 存储构造出的平面的方程系数
        point_selected_surf[i] = false; // 再次赋值为false，用于后续计算是都为平面点
        // 用邻近点拟合平面方程，系数保存到pabcd中
        if (esti_plane(pabcd, points_near, 0.1f)) // 平面拟合成功
        {
            // 计算世界坐标系下的点到平面的距离
            float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y + pabcd(2) * point_world.z + pabcd(3);

            // 判断该点是否是平面点
            float s = 1 - 0.9 * fabs(pd2) / sqrt(p_body.norm());
            if (s > 0.9) // 该点是平面点
            {
                point_selected_surf[i] = true; // 标志为true
                normvec->points[i].x = pabcd(0); // 存储该平面的法向量
                normvec->points[i].y = pabcd(1);
                normvec->points[i].z = pabcd(2);
                normvec->points[i].intensity = pd2; // 将点到平面的距离作为intensity
                res_last[i] = abs(pd2); // 将所有平面点的残差都存储到res_last中
            }
        }
    }
    
    effct_feat_num = 0; // 统计有效特征点数量——就是平面点数量

    // 存储所有平面点在LiDAR系的坐标、对应世界平面的法向量
    // 求和残差
    // 统计有效平面点数量
    for (int i = 0; i < feats_down_size; i++)
    {
        if (point_selected_surf[i])
        {
            // 这里的body其实就是指得LiDAR
            laserCloudOri->points[effct_feat_num] = feats_down_body->points[i]; 
            corr_normvect->points[effct_feat_num] = normvec->points[i];
            total_residual += res_last[i];
            effct_feat_num ++;
        }
    }

    // 没有有效的平面点，直接返回
    if (effct_feat_num < 1)
    {
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }

    res_mean_last = total_residual / effct_feat_num; // 残差均值
    match_time  += omp_get_wtime() - match_start;
    double solve_start_  = omp_get_wtime();
    
    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    // 计算观测值在delta_x=0处的雅可比矩阵h_x
    // TODO：12维——这里只计算了h_x中相对于delta_t，delta_theta以及外参的平移和旋转的偏导数
    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12); //23
    ekfom_data.h.resize(effct_feat_num);

    for (int i = 0; i < effct_feat_num; i++)
    {
        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(laser_p.x, laser_p.y, laser_p.z);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
        const PointType &norm_p = corr_normvect->points[i];
        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en) // 如果估计外参，那么需要计算相对于外参状态误差量的偏导数
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_p.x, norm_p.y, norm_p.z, VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }

        /*** Measuremnt: distance to the closest surface/corner ***/
        // 观测向量h中的每个值就是对应面点到平面的距离
        ekfom_data.h(i) = -norm_p.intensity;
    }
    solve_time += omp_get_wtime() - solve_start_;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    // Step 1 读取配置参数，如果没有就使用默认参数
    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);
    nh.param<int>("max_iteration",NUM_MAX_ITERATIONS,4);
    nh.param<string>("map_file_path",map_file_path,"");
    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner",filter_size_corner_min,0.5);
    nh.param<double>("filter_size_surf",filter_size_surf_min,0.5);
    nh.param<double>("filter_size_map",filter_size_map_min,0.5);
    nh.param<double>("cube_side_length",cube_len,200);
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);
    nh.param<double>("mapping/fov_degree",fov_deg,180);
    nh.param<double>("mapping/gyr_cov",gyr_cov,0.1);
    nh.param<double>("mapping/acc_cov",acc_cov,0.1);
    nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);
    nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01); // 雷达最小有效距离，也可以说是盲区范围
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA); // 雷达类型，默认是AVIA类型的雷达
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16); // 雷达线数，默认是16线
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US); // 时间单位
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10); // 雷达扫描帧率
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2); // 等间隔降采样
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false); // 是否进行特征提取
    nh.param<bool>("runtime_pos_log_enable", runtime_pos_log, 0);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en, false);
    nh.param<int>("pcd_save/interval", pcd_save_interval, -1);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init"; // 将坐标系定义为：camera_init

    /*** variables definition ***/
    // Step 2 变量定义及设置
    // effect_feat_num = 有效的特征点数量；frame_num = 雷达总帧数
    int effect_feat_num = 0, frame_num = 0;
    // aver_time_consu = 平均每帧处理时间
    // aver_time_icp = 平均每帧ICP处理时间
    // aver_time_match = 平均每帧点云匹配处理时间
    // aver_time_incre = 平均每帧kd-tree增量处理时间
    // aver_time_solve = 平均每帧状态量求解时间
    // aver_time_const_H_time = 平均每帧H计算时间
    double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_const_H_time = 0;
    // flg_EKF_converged = EKF估计是否收敛
    // EKF_stop_flg = 是否停止EKF估计
    bool flg_EKF_converged, EKF_stop_flg = 0;
    
    // 总视野和半视野场大小对应的cos值
    FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0); // 控制在 0～179.9度内
    HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);
   
    // _featsArray初始化
    _featsArray.reset(new PointCloudXYZI()); // ikd-tree中，map需要移除的点云序列

    // 将数组point_selected_surf内元素的值全部初始化为true ，用于标记平面点
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    // 残差，点到平面的距离平方，这里初始化为-1000.0，数组res_last用于平面拟合中
    memset(res_last, -1000.0f, sizeof(res_last)); 
    // VoxelGrid滤波器参数，即进行滤波时的创建的体素边长为filter_size_surf_min
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    // VoxelGrid滤波器参数，即进行滤波时的创建的体素边长为filter_size_map_min
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    memset(point_selected_surf, true, sizeof(point_selected_surf)); // 重复操作了
    memset(res_last, -1000.0f, sizeof(res_last));

    // 雷达和IMU外参设置，以及测量以及偏置的协方差初始化
    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    // 将double形数组epsi中每个元数都赋值为0.001
    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    // 关键！！！后续将用于状态前后、后向传播需要的函数（get_f, df_dx, df_dw, h_share_model）通过init_dyn_share初始化为esekf类中的相关变量
    kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(),"w");

    ofstream fout_pre, fout_out, fout_dbg;
    fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"),ios::out);
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"),ios::out);
    fout_dbg.open(DEBUG_FILE_DIR("dbg.txt"),ios::out);
    if (fout_pre && fout_out)
        cout << "~~~~"<<ROOT_DIR<<" file opened" << endl;
    else
        cout << "~~~~"<<ROOT_DIR<<" doesn't exist" << endl;

    /*** ROS subscribe initialization ***/
    // Step 3 雷达、IMU数据订阅以及预处理，并初始化相关发布节点
    // 根据雷达类型选取callback()函数
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);

    // IMU数据的callback()函数——进行IMU数据的处理：IMU积分
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);

    // 初始化ros节点，发布相关信息
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> 
            ("/Odometry", 100000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path> 
            ("/path", 100000);

    //------------------------------------------------------------------------------------------------------
    // Step 4 将所有数据用于状态估计
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000); // 主循环运行的频率至少是5000Hz，即时间不少于0.0002秒
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit) break;
        ros::spinOnce();
        // Step 4.1 主要是将一帧激光点云，以及这一帧点云扫描期间的imu数据 一起打包放到MeasureGroup里。
        if(sync_packages(Measures)) 
        {
            // Step 4.2 第一帧数据，记录时间戳，然后跳过即可
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }

            double t0,t1,t2,t3,t4,t5,match_start, solve_start, svd_time;

            match_time = 0;
            kdtree_search_time = 0.0;
            solve_time = 0;
            solve_const_H_time = 0;
            svd_time   = 0;
            t0 = omp_get_wtime();
            
            // Step 4.3 IMU初始化和状态量的前向传播，后向传播，并完成点云去畸变
            p_imu->Process(Measures, kf, feats_undistort); // feats_undistort ：去畸变后的点云特征
            state_point = kf.get_x(); // 取出当前帧结束时间的系统状态——应该是在当前IMU系下
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I; // 当前帧相对于World系下的LiDAR坐标

            // 如果去畸变后的点云特征是空的，直接退出
            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            // 判断是否完成初始化，成功——当前测量值Measures中雷达的结束时间 和 第一帧雷达数据的开始时间 大于 规定的初始化时间
            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            /*** Segment the map in lidar FOV ***/
            // Step 4.4 地图更新
            lasermap_fov_segment();

            /*** downsample the feature points in a scan ***/
            // 当前点降采样
            downSizeFilterSurf.setInputCloud(feats_undistort); 
            downSizeFilterSurf.filter(*feats_down_body); // feats_undistort降采样滤波后，得到feats_down_body
            t1 = omp_get_wtime();
            feats_down_size = feats_down_body->points.size(); // 降采样后的点云数量
            /*** initialize the map kdtree ***/
            // 初始化kd-tree，构建kd-tree
            if(ikdtree.Root_Node == nullptr)
            {
                // 构建kd-tree的点云必须大于5，否则跳过
                if(feats_down_size > 5)
                {
                    // 设置kd-tree降采样参数
                    ikdtree.set_downsample_param(filter_size_map_min); // filter_size_map_min = 0.5(米)
                    feats_down_world->resize(feats_down_size); // 预分配空间，存储在世界系下的点云位置
                    // 遍历所有降采样后的去畸变点云，将其从Lidar系转到World系下，并在pointBodyToWorld中保存world系下的位置
                    for(int i = 0; i < feats_down_size; i++)
                    {
                        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                    }
                    // 将降采样后的在世界坐标系下的点云用于构建kd-tree树
                    ikdtree.Build(feats_down_world->points); // 构建kd-tree树
                }
                continue;
            }
            int featsFromMapNum = ikdtree.validnum(); // 获取kd-tree中有效的节点数量
            kdtree_size_st = ikdtree.size(); // 获取kd-tree中所有节点数量
            
            // cout<<"[ mapping ]: In num: "<<feats_undistort->points.size()<<" downsamp "<<feats_down_size<<" Map num: "<<featsFromMapNum<<"effect num:"<<effct_feat_num<<endl;

            /*** ICP and iterated Kalman filter update ***/
            // 如果降采样后的点云数量小于 5 不进行迭代状态更新，直接跳过当前帧的数据
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }
            
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            V3D ext_euler = SO3ToEuler(state_point.offset_R_L_I); // 将外参的旋转矩阵转换为欧拉角
            fout_pre<<setw(20)<<Measures.lidar_beg_time - first_lidar_time<<" "<<euler_cur.transpose()<<" "<< state_point.pos.transpose()<<" "<<ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<< " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<< endl;

            if(0) // If you need to see map point, change to "if(1)"
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            pointSearchInd_surf.resize(feats_down_size);
            Nearest_Points.resize(feats_down_size);
            int  rematch_num = 0;
            bool nearest_search_en = true; // 最近点搜索

            t2 = omp_get_wtime();
            
            /*** iterated state estimation ***/
            // 迭代状态估计
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            // ！！！关键点！！！迭代卡尔曼滤波更新，更新地图信息
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, solve_H_time);
            state_point = kf.get_x(); // 此时的x_是当前最优估计值
            euler_cur = SO3ToEuler(state_point.rot); // 取出其中的旋转矩阵，并转为欧拉角
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I; // TODO：这是要干嘛
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];

            double t_update_end = omp_get_wtime();

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);

            /*** add the feature points to map kdtree ***/
            // 在局部地图 kdtree中加点
            t3 = omp_get_wtime();
            map_incremental();
            t5 = omp_get_wtime();
            
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en || pcd_save_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
            // publish_effect_world(pubLaserCloudEffect);
            // publish_map(pubLaserCloudMap);

            /*** Debug variables ***/
            if (runtime_pos_log)
            {
                frame_num ++;
                kdtree_size_end = ikdtree.size();
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                aver_time_icp = aver_time_icp * (frame_num - 1)/frame_num + (t_update_end - t_update_start) / frame_num;
                aver_time_match = aver_time_match * (frame_num - 1)/frame_num + (match_time)/frame_num;
                aver_time_incre = aver_time_incre * (frame_num - 1)/frame_num + (kdtree_incremental_time)/frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
                aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = kdtree_incremental_time;
                s_plot4[time_log_counter] = kdtree_search_time;
                s_plot5[time_log_counter] = kdtree_delete_counter;
                s_plot6[time_log_counter] = kdtree_delete_time;
                s_plot7[time_log_counter] = kdtree_size_st;
                s_plot8[time_log_counter] = kdtree_size_end;
                s_plot9[time_log_counter] = aver_time_consu;
                s_plot10[time_log_counter] = add_point_size;
                time_log_counter ++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f construct H: %0.6f \n",t1-t0,aver_time_match,aver_time_solve,t3-t1,t5-t3,aver_time_consu,aver_time_icp, aver_time_const_H_time);
                ext_euler = SO3ToEuler(state_point.offset_R_L_I);
                fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " " << euler_cur.transpose() << " " << state_point.pos.transpose()<< " " << ext_euler.transpose() << " "<<state_point.offset_T_L_I.transpose()<<" "<< state_point.vel.transpose() \
                <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
                dump_lio_state_to_log(fp);
            }
        }

        status = ros::ok();
        rate.sleep();
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. pcd save will largely influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en)
    {
        string file_name = string("scans.pcd");
        string all_points_dir(string(string(ROOT_DIR) + "PCD/") + file_name);
        pcl::PCDWriter pcd_writer;
        cout << "current scan saved to /PCD/" << file_name<<endl;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
    }

    fout_out.close();
    fout_pre.close();

    if (runtime_pos_log)
    {
        vector<double> t, s_vec, s_vec2, s_vec3, s_vec4, s_vec5, s_vec6, s_vec7;    
        FILE *fp2;
        string log_dir = root_dir + "/Log/fast_lio_time_log.csv";
        fp2 = fopen(log_dir.c_str(),"w");
        fprintf(fp2,"time_stamp, total time, scan point size, incremental time, search time, delete size, delete time, tree size st, tree size end, add point size, preprocess time\n");
        for (int i = 0;i<time_log_counter; i++){
            fprintf(fp2,"%0.8f,%0.8f,%d,%0.8f,%0.8f,%d,%0.8f,%d,%d,%d,%0.8f\n",T1[i],s_plot[i],int(s_plot2[i]),s_plot3[i],s_plot4[i],int(s_plot5[i]),s_plot6[i],int(s_plot7[i]),int(s_plot8[i]), int(s_plot10[i]), s_plot11[i]);
            t.push_back(T1[i]);
            s_vec.push_back(s_plot9[i]);
            s_vec2.push_back(s_plot3[i] + s_plot6[i]);
            s_vec3.push_back(s_plot4[i]);
            s_vec5.push_back(s_plot[i]);
        }
        fclose(fp2);
    }

    return 0;
}
