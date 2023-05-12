#include <cmath>
#include <math.h>
#include <deque>
#include <mutex>
#include <thread>
#include <fstream>
#include <csignal>
#include <ros/ros.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <common_lib.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <condition_variable>
#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <tf/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/Vector3.h>
#include "use-ikfom.hpp"

/// *************Preconfiguration

#define MAX_INI_COUNT (10)

// 判断时间戳大小
const bool time_list(PointType &x, PointType &y) {return (x.curvature < y.curvature);};

/// *************IMU Process and undistortion = IMU积分(IMU前向传播) + 点云去畸变（后向传播）
class ImuProcess
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess(); // 构造函数
  ~ImuProcess();
  
  void Reset();
  void Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void set_extrinsic(const V3D &transl, const M3D &rot);
  void set_extrinsic(const V3D &transl);
  void set_extrinsic(const MD(4,4) &T);
  void set_gyr_cov(const V3D &scaler);
  void set_acc_cov(const V3D &scaler);
  void set_gyr_bias_cov(const V3D &b_g);
  void set_acc_bias_cov(const V3D &b_a);
  Eigen::Matrix<double, 12, 12> Q; 
  void Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr pcl_un_);

  ofstream fout_imu; // IMU相关文件流输出
  // TODO ：注意加scale和不加scale后续的区别
  V3D cov_acc;              // 加速度计测量协方差
  V3D cov_gyr;              // 陀螺仪测量协方差
  V3D cov_acc_scale; // 加速度计测量协方差—— TODO：这里的sacle指得是数据归一化之后的协方差吗？
  V3D cov_gyr_scale; // 陀螺仪测量协方差
  V3D cov_bias_gyr;   // 陀螺仪偏置协方差
  V3D cov_bias_acc;   // 加速度计偏置协方差
  double first_lidar_time; // 第一帧雷达数据时间戳

 private:
  void IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_in_out);

  PointCloudXYZI::Ptr cur_pcl_un_;  // 去畸变后的点云
  sensor_msgs::ImuConstPtr last_imu_; // 上一帧IMU数据
  deque<sensor_msgs::ImuConstPtr> v_imu_; // IMU数据队列
  vector<Pose6D> IMUpose; // IMU积分得到的位姿(R t)
  vector<M3D>    v_rot_pcl_; // TODO
  M3D Lidar_R_wrt_IMU;  // Lidar系到IMU系的旋转
  V3D Lidar_T_wrt_IMU;    // Lidar系到IMU系的平移
  V3D mean_acc; // 加速度计测量的均值
  V3D mean_gyr; // 陀螺仪测量的均值
  V3D angvel_last; // 上一帧的角速度
  V3D acc_s_last;   // 上一帧的加速度
  double start_timestamp_;  // 
  double last_lidar_end_time_;
  int    init_iter_num = 1;
  bool   b_first_frame_ = true; // 是否是第一帧
  bool   imu_need_init_ = true; // 是否需要进行IMU初始化
};

// ImuProcess构造函数
// b_first_frame_ = true 是第一帧数据，之后的为false
// imu_need_init_ = true，完成IMU初始化之后为false
// start_timestamp_ 开始时间戳，初始化为-1
ImuProcess::ImuProcess()
    : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1)
{
  init_iter_num = 1; // 迭代次数
  Q = process_noise_cov(); // 完成加速度计测量噪声、陀螺仪测量噪声、加速度计偏置噪声、陀螺仪偏置噪声的协方差初始化
  cov_acc       = V3D(0.1, 0.1, 0.1);
  cov_gyr       = V3D(0.1, 0.1, 0.1);
  cov_bias_gyr  = V3D(0.0001, 0.0001, 0.0001);
  cov_bias_acc  = V3D(0.0001, 0.0001, 0.0001);
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last     = Zero3d; // 初始化为0
  Lidar_T_wrt_IMU = Zero3d; // 初始化为0
  Lidar_R_wrt_IMU = Eye3d; // 初始化为单位矩阵
  last_imu_.reset(new sensor_msgs::Imu()); // 初始化为sensor_msgs::Imu()
}

// 析构
ImuProcess::~ImuProcess() {}

// IMU类重置
void ImuProcess::Reset() 
{
  // ROS_WARN("Reset ImuProcess");
  mean_acc      = V3D(0, 0, -1.0);
  mean_gyr      = V3D(0, 0, 0);
  angvel_last       = Zero3d;
  imu_need_init_    = true;
  start_timestamp_  = -1;
  init_iter_num     = 1;
  v_imu_.clear(); // IMU数据队列清空
  IMUpose.clear(); // IMU位姿清空
  last_imu_.reset(new sensor_msgs::Imu()); // 上一帧IMU指针重新初始化
  cur_pcl_un_.reset(new PointCloudXYZI()); // 当前去畸变后的点云指针重新初始化
}

// 设置Lidar和IMU之间的外参
// 转入的是变换矩阵
void ImuProcess::set_extrinsic(const MD(4,4) &T)
{
  Lidar_T_wrt_IMU = T.block<3,1>(0,3);
  Lidar_R_wrt_IMU = T.block<3,3>(0,0);
}

// 仅有平移
void ImuProcess::set_extrinsic(const V3D &transl)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU.setIdentity();
}

// 平移 + 旋转
void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot)
{
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}

// 设置陀螺仪测量协方差
void ImuProcess::set_gyr_cov(const V3D &scaler)
{
  cov_gyr_scale = scaler;
}

// 设置加速度计测量协方差
void ImuProcess::set_acc_cov(const V3D &scaler)
{
  cov_acc_scale = scaler;
}

// 设置陀螺仪偏置协方差
void ImuProcess::set_gyr_bias_cov(const V3D &b_g)
{
  cov_bias_gyr = b_g;
}

// 设置加速度计偏置协方差
void ImuProcess::set_acc_bias_cov(const V3D &b_a)
{
  cov_bias_acc = b_a;
}

/// @brief IMU初始化：完成状态量x_和协方差矩阵P_的初始化
/// @param meas 包含当前帧预处理之后的点云，以及当前点云之间的所有IMU数据
/// @param kf_state 系统状态量
/// @param N 
void ImuProcess::IMU_init(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N)
{
  /** 1. initializing the gravity, gyro bias, acc and gyro covariance 初始化重力加速度，陀螺仪偏置，加速度计和陀螺仪的协方差
   ** 2. normalize the acceleration measurenments to unit gravity **/ // 将加速度测量值归一化为单位重力加速度
  
  V3D cur_acc, cur_gyr; // 存储当前的线加速度和角速度
  
  // Step 1 如果是第一帧，完成参数重置，并且记录第一帧相关数据
  if (b_first_frame_)
  {
    Reset(); // 参数重置
    N = 1;
    b_first_frame_ = false;
    const auto &imu_acc = meas.imu.front()->linear_acceleration;
    const auto &gyr_acc = meas.imu.front()->angular_velocity;
    mean_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    mean_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;
    first_lidar_time = meas.lidar_beg_time;
  }
  
  // Step 2 遍历所有IMU数据，取加速度和角加速度的均值，并计算加速度和陀螺仪的协方差
  for (const auto &imu : meas.imu)
  {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
    cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

    // 关于均值和协方差的递推过程，可参考https://zhuanlan.zhihu.com/p/445729443
    mean_acc      += (cur_acc - mean_acc) / N;
    mean_gyr      += (cur_gyr - mean_gyr) / N;

    cov_acc = cov_acc * (N - 1.0) / N + (cur_acc - mean_acc).cwiseProduct(cur_acc - mean_acc) * (N - 1.0) / (N * N);
    cov_gyr = cov_gyr * (N - 1.0) / N + (cur_gyr - mean_gyr).cwiseProduct(cur_gyr - mean_gyr) * (N - 1.0) / (N * N); 

    // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;

    N ++; // 记录有多少个IMU数据
  }

  // Step 3 状态量x_和协方差P_的初始化
  // 获取状态量
  state_ikfom init_state = kf_state.get_x();
  // 根据当前加速度测量均值获取的单位重力，求出SO2旋转类型的重力加速度（应该就是加速度均值为SO2的重力矢量提供一个长度）
  init_state.grav = S2(- mean_acc / mean_acc.norm() * G_m_s2);
  
  //state_inout.rot = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
  init_state.bg  = mean_gyr;
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;
  init_state.offset_R_L_I = Lidar_R_wrt_IMU;
  kf_state.change_x(init_state);

  // TODO：P中各位置上的物理含义搞清楚
  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
  init_P.setIdentity();
  // 平移和旋转的协方差初始化为0.00001
  init_P(6,6) = init_P(7,7) = init_P(8,8) = 0.00001;
  // 速度和位姿的协方差置初始化为0.00001
  init_P(9,9) = init_P(10,10) = init_P(11,11) = 0.00001;
  // 重力和姿态的协方差初始化为0.0001
  init_P(15,15) = init_P(16,16) = init_P(17,17) = 0.0001;
  // 重力和姿态的协方差初始化为0.0001
  init_P(18,18) = init_P(19,19) = init_P(20,20) = 0.001;
  // lidar和imu外参位移量的协方差初始化为0.00001
  init_P(21,21) = init_P(22,22) = 0.00001; 
  kf_state.change_P(init_P);
  last_imu_ = meas.imu.back();

}

/// @brief 完成IMU前后传播，状态后向传播和点云去畸变
/// @param[in] meas 当前帧预处理之后的点云以及当前帧内的所有IMU测量值
/// @param[in out] kf_state 系统状态
/// @param[out] pcl_out 去畸变后的点云
void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI &pcl_out)
{
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  // Step 1 将上一帧最后一个IMU添加到当前帧头的IMU中。
  auto v_imu = meas.imu; // 取出当前IMU的vector
  v_imu.push_front(last_imu_); // 将上一帧的最后一个IMU数据放进当前IMU容器的头部——将上一帧和当前帧的IMU数据联合，在上一帧和当前帧状态之间起到一个过渡的作用
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec(); // 当前IMU容器中第一个IMU数据的时间戳（实际上是上一帧IMU的最后一个数据的时间戳）
  const double &imu_end_time = v_imu.back()->header.stamp.toSec(); // 当前IMU容器中最后一个IMU数据的时间戳
  const double &pcl_beg_time = meas.lidar_beg_time; // 当前点云帧的开始时间
  const double &pcl_end_time = meas.lidar_end_time; // 当前点云帧的结束时间
  
  /*** sort point clouds by offset time ***/
  // Step 2 根据点云获取时间，从小到大进行排序
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;

  /*** Initialize IMU pose ***/
  // Step 3 使用上一帧KF估计的状态初始化当前IMU的pose
  state_ikfom imu_state = kf_state.get_x(); // 将上一次KF估计的后验状态作为本次IMU预测的初始状态
  IMUpose.clear(); // vector<Pose6D> IMUpose 上一帧的IMUpose清零
  // 将初始状态加入IMUpose中,包含有时间间隔，上一帧加速度，上一帧角速度，上一帧速度，上一帧位置，上一帧旋转矩阵
  IMUpose.push_back(set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));

  /*** forward propagation at each imu point ***/
  // Step 4 在当前帧的IMU数据间进行前向传播(IMU积分)
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;

  double dt = 0;

  input_ikfom in; // 输入状态参数——IMU的线加速度和角速度

  // 遍历当前帧的所有IMU数据
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++)
  {
    auto &&head = *(it_imu); // 当前时刻的IMU数据
    auto &&tail = *(it_imu + 1); // 下一时刻的IMU数据
    
    // 下一时刻的IMU数据 小于 上一时刻的LiDAR结束时间，跳过当前IMU数据
    if (tail->header.stamp.toSec() < last_lidar_end_time_)    continue;
    
    // 当前时刻和下一时刻IMU数据的中值积分
    angvel_avr<<0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr   <<0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    // fout_imu << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose() << " " << acc_avr.transpose() << endl;
    // 根据G_m_s2 / mean_acc.norm()比例对加速度进行调节
    acc_avr     = acc_avr * G_m_s2 / mean_acc.norm(); // - state_inout.ba;

    // 获取当前和下一个IMU时刻之间的时间间隔
    if(head->header.stamp.toSec() < last_lidar_end_time_)
    {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
      // dt = tail->header.stamp.toSec() - pcl_beg_time;
    }
    else
    {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }
    
    // 用当前和下一时刻的IMU数据中值更新
    in.acc = acc_avr;
    in.gyro = angvel_avr;
    // 上一帧的协方差作为当前噪声协方差的初值
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    // 每两个IMU时刻之间都进行状态的前向传播
    kf_state.predict(dt, Q, in); // 迭代误差状态卡尔曼传播，得到前向传播后的x_，以及相应的状态转移矩阵F_x1和协方差矩阵P_

    /* save the poses at each IMU measurements */
    // 保存每个时刻的IMU位姿
    imu_state = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);
    for(int i=0; i<3; i++)
    {
      acc_s_last[i] += imu_state.grav[i];
    }
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    IMUpose.push_back(set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  // Step 5 计算帧尾的状态
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt = note * (pcl_end_time - imu_end_time); // 时间间隔
  kf_state.predict(dt, Q, in); // 迭代误差状态卡尔曼传播，得到前向传播后的x_，以及相应的状态转移矩阵F_x1和协方差矩阵P_
  
  imu_state = kf_state.get_x(); // 取出完成前向传播后的系统状态
  last_imu_ = meas.imu.back(); // 当前阵中最后一个IMU数据
  last_lidar_end_time_ = pcl_end_time; // 当前帧雷达数据结束时间

  /*** undistort each lidar point (backward propagation) ***/
  // Step 6 点云去畸变（也称为：后向传播）
  if (pcl_out.points.begin() == pcl_out.points.end()) return; // 没有可用点云信息，退出
  auto it_pcl = pcl_out.points.end() - 1; // 点云数量
  // 遍历每一个IMUPose，从尾到头进行遍历
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--)
  {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu<<MAT_FROM_ARRAY(head->rot); // 上一时刻IMU旋转矩阵
    // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
    vel_imu<<VEC_FROM_ARRAY(head->vel); // 前一时刻IMU速度
    pos_imu<<VEC_FROM_ARRAY(head->pos); // 前一时刻IMU位置
    acc_imu<<VEC_FROM_ARRAY(tail->acc); // 当前时刻IMU加速度
    angvel_avr<<VEC_FROM_ARRAY(tail->gyr); // 当前时刻IMU角速度

    // 遍历在head - tail 时间段内的点云，将其补偿到帧尾
    // 注意：在两个IMU时刻之间进行点云去畸变，所以理论上当前点云时间应该迟于上一个时刻的IMU数据
    for(; it_pcl->curvature / double(1000) > head->offset_time; it_pcl --)
    {
      // 没有进行额外的判断，应该是认为当前点云的时间就是在上一时刻IMU之后 
      dt = it_pcl->curvature / double(1000) - head->offset_time; // 当前点云和上一时刻IMU时间戳差值

      /* Transform to the 'end' frame, using only the rotation 仅使用IMU积分的旋转将其转到当前帧的帧尾
       * Note: Compensation direction is INVERSE of Frame's moving direction 补偿方向与帧的移动方向相反
       * So if we want to compensate a point at timestamp-i to the frame-e 所以，如果想要将时刻i的点云补偿到帧尾
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */ 
      M3D R_i(R_imu * Exp(angvel_avr, dt)); // 相对于当前点云时刻的IMU旋转矩阵
      
      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z); // 当前点云位置
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos); // 相对于当前点云时刻的IMU平移 - 帧尾相对于世界坐标下的平移
      // 最后还是转到了当前帧结束时刻的Lidar系下
      // 第一步：从LiDAR系转到IMU系 imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I
      // 第二步：转到当前时刻i对应的IMU系下 (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei)
      // 第三步：仅利用状态量在帧尾的旋转矩阵将其从第i时刻IMU系转到帧尾时刻的IMU系
      // 第四步：再次使用外参，从帧尾对应的IMU系转到帧尾对应的LiDAR系
      V3D P_compensate = imu_state.offset_R_L_I.conjugate() * (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) - imu_state.offset_T_L_I);// not accurate! 
      
      // save Undistorted points and their rotation
      // 保存运动补偿后的点云在LiDAR系下的3D位置
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}

/// @brief IMU初始化和状态量的前向传播，后向传播，并完成点云去畸变
/// @param[in] meas 当前帧点云和当前帧间之间所有可用的IMU时刻
/// @param[out] kf_state 系统状态
/// @param[out] cur_pcl_un_ 去畸变的点云
void ImuProcess::Process(const MeasureGroup &meas,  esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, PointCloudXYZI::Ptr cur_pcl_un_)
{
  double t1,t2,t3;
  t1 = omp_get_wtime(); // 记录当前时间

  // 如果没有IMU数据，直接返回
  if(meas.imu.empty()) {return;};
  ROS_ASSERT(meas.lidar != nullptr);
  
  // 第一帧进行IMU初始化后，直接返回
  // 后续帧完成状态量的前向传播，后向传播，以及点云去畸变
  if (imu_need_init_)
  {
    /// The very first lidar frame
    // Step 1 IMU初始化，完成状态量x_和协方差P_的初始化，其中init_iter_num记录过程中使用的IMU测量数据总量
    IMU_init(meas, kf_state, init_iter_num);

    imu_need_init_ = true;
    
    last_imu_   = meas.imu.back();

    state_ikfom imu_state = kf_state.get_x();
    if (init_iter_num > MAX_INI_COUNT) // 如果使用的IMU测量数据总量超过最大值
    {
      cov_acc *= pow(G_m_s2 / mean_acc.norm(), 2); // 在IMU初始化的基础上，乘以缩放系数
      imu_need_init_ = false; // 还需初始化

      // 如果这里需要将cov_acc/cov_gyr赋值为默认值，干嘛之前还缩放呢？？？
      cov_acc = cov_acc_scale;
      cov_gyr = cov_gyr_scale;
      ROS_INFO("IMU Initial Done");
      // ROS_INFO("IMU Initial Done: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f",\
      //          imu_state.grav[0], imu_state.grav[1], imu_state.grav[2], mean_acc.norm(), cov_bias_gyr[0], cov_bias_gyr[1], cov_bias_gyr[2], cov_acc[0], cov_acc[1], cov_acc[2], cov_gyr[0], cov_gyr[1], cov_gyr[2]);
      fout_imu.open(DEBUG_FILE_DIR("imu.txt"),ios::out);
    }

    return;
  }

  // Step 2 状态量的前向传播，后向传播，并完成点云去畸变
  UndistortPcl(meas, kf_state, *cur_pcl_un_);

  t2 = omp_get_wtime();
  t3 = omp_get_wtime();
  
  // cout<<"[ IMU Process ]: Time: "<<t3 - t1<<endl;
}
