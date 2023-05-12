#include "preprocess.h"

#define RETURN0     0x00
#define RETURN0AND1 0x10

Preprocess::Preprocess()
  :feature_enabled(0), lidar_type(AVIA), blind(0.01), point_filter_num(1)
{
  inf_bound = 10;
  N_SCANS   = 6;
  SCAN_RATE = 10;
  group_size = 8;
  disA = 0.01;
  // disA = 0.1; // B? 应该是disB = 0.1
  disB = 0.1;
  p2l_ratio = 225;
  limit_maxmid =6.25;
  limit_midmin =6.25;
  limit_maxmin = 3.24;
  jump_up_limit = 170.0;
  jump_down_limit = 8.0;
  cos160 = 160.0;
  edgea = 2;
  edgeb = 0.1;
  smallp_intersect = 172.5;
  smallp_ratio = 1.2;
  given_offset_time = false;

  jump_up_limit = cos(jump_up_limit/180*M_PI);
  jump_down_limit = cos(jump_down_limit/180*M_PI);
  cos160 = cos(cos160/180*M_PI);
  smallp_intersect = cos(smallp_intersect/180*M_PI);
}

Preprocess::~Preprocess() {}

void Preprocess::set(bool feat_en, int lid_type, double bld, int pfilt_num)
{
  feature_enabled = feat_en;
  lidar_type = lid_type;
  blind = bld;
  point_filter_num = pfilt_num;
}

/// @brief 进行livox雷达数据处理
/// @param msg 接收到未经处理的点云数据
/// @param pcl_out 输出预处理之后的点云数据
void Preprocess::process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{
  avia_handler(msg);    // 预处理雷达数据
  *pcl_out = pl_surf;   // TODO：仅保存面点？？？——实际应用中，面点的数量是远远大于边缘点的
}

/// @brief 进行机械激光雷达数据预处理
/// @param msg 接收到未经处理的点云数据
/// @param pcl_out 输出预处理之后的点云数据
void Preprocess::process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out)
{
  // 根据时间单位，设置time_unit_scale的值
  switch (time_unit)
  {
    case SEC:
      time_unit_scale = 1.e3f;
      break;
    case MS:
      time_unit_scale = 1.f;
      break;
    case US:
      time_unit_scale = 1.e-3f;
      break;
    case NS:
      time_unit_scale = 1.e-6f;
      break;
    default:
      time_unit_scale = 1.f;
      break;
  }

  // 根据雷达线数选择处理程序
  switch (lidar_type)
  {
  case OUST64:
    oust64_handler(msg);
    break;

  case VELO16:
    velodyne_handler(msg);
    break;
  
  default:
    printf("Error LiDAR Type");
    break;
  }
  *pcl_out = pl_surf;
}

/// @brief 进行livox雷达数据预处理
/// @param msg 输入未经处理的雷达数据
void Preprocess::avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
  // Step 1 将存储点云以及特征点的buff都清空，避免造成干扰
  pl_surf.clear();
  pl_corn.clear();
  pl_full.clear();
  double t1 = omp_get_wtime(); // 记录开始进行数据预处理的开始时间
  int plsize = msg->point_num;    // 该帧中的所有未处理的点云数量
  // cout<<"plsie: "<<plsize<<endl;

  // Step 2 根据处理之前的点云数量为相关buff预分配空间，空间肯定会大一些
  pl_corn.reserve(plsize);
  pl_surf.reserve(plsize);
  pl_full.resize(plsize);

  // Step 3 根据雷达线数，清空之前的旧数据，并分配存储每条SCAN点云的空间
  for(int i=0; i<N_SCANS; i++)
  {
    pl_buff[i].clear();
    pl_buff[i].reserve(plsize);
  }

  uint valid_num = 0; // 记录有效点云数量
  
  // Step 4 是否进行特征提取
  // 一般在FAST-LIO2之后，默认是不进行特征提取的
  if (feature_enabled) // Step 4.1 如果进行特征提取
  {
    // 遍历所有点云，根据scan数分别存储到相应的pl_full中
    // 注意这里从i = 1开始
    for(uint i=1; i<plsize; i++)
    {
      // Step 4.1.1 判断当前点云的合法性
      // 当前点云所在线数 < 总线数
      // 当前点云的回波次序为0或者1 
      // 相关参考：https://www.163.com/dy/article/GBKVGBI60514GCU0.html 和 https://zhuanlan.zhihu.com/p/461364648
      if((msg->points[i].line < N_SCANS) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
      {
        // Step 4.1.2 提出当前点云的位置、反射率以及曲率（曲率实际上是当前SCAN上的每个激光点相对于起始扫描扫描时间的时间差）
        pl_full[i].x = msg->points[i].x;
        pl_full[i].y = msg->points[i].y;
        pl_full[i].z = msg->points[i].z;
        pl_full[i].intensity = msg->points[i].reflectivity;
        pl_full[i].curvature = msg->points[i].offset_time / float(1000000); //use curvature as time of each laser points 使用曲率作为每个激光点的时间

        bool is_new = false;
        // Step 4.1.3 如果当前点和上一个点在任意方向(x y z)上的距离差 > 1e-7，认为是有效的点，并加入到索引为scan的pl_full中
        // 因为是从i = 1开始，所以不用担心i-1的合法性
        if((abs(pl_full[i].x - pl_full[i-1].x) > 1e-7) 
            || (abs(pl_full[i].y - pl_full[i-1].y) > 1e-7)
            || (abs(pl_full[i].z - pl_full[i-1].z) > 1e-7))
        {
          pl_buff[msg->points[i].line].push_back(pl_full[i]);
        }
      }
    }
    
    // Step 4.1.4 设置相关参数
    static int count = 0;
    static double time = 0.0;
    count ++;
    double t0 = omp_get_wtime();
    // Step 4.1.5 遍历所有scan上的点云数据
    for(int j=0; j<N_SCANS; j++)
    {
      if(pl_buff[j].size() <= 5) continue; // 如果当前scan上的点数量 < 5，跳过该条scan
      pcl::PointCloud<PointType> &pl = pl_buff[j]; // 获取当前scan中的所有点云，可通过pl进行索引和修改
      plsize = pl.size(); // 当前scan中的点云数量
      vector<orgtype> &types = typess[j]; // 获取当前scan中的所有点云状态，可通过types进行索引和修改
       // 注意：这里orgtype在初始化的时候将相关点云状态初始化为正常
      types.clear(); // 清空旧数据
      types.resize(plsize); // 预分配空间
      plsize--; // 这里plsize-- 实际上是考虑下列遍历过程中pl[i + 1]的问题，避免最后一个点的遍历超出索引范围
      // 遍历当前scan中的所有点，计算当前点云在xy平面上的距离，以及当前点云距离下一个点云的距离
      for(uint i=0; i<plsize; i++)
      {
        types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
        vx = pl[i].x - pl[i + 1].x;
        vy = pl[i].y - pl[i + 1].y;
        vz = pl[i].z - pl[i + 1].z;
        types[i].dista = sqrt(vx * vx + vy * vy + vz * vz);
      }
      // 最后一个点云在xy平面上的距离
      types[plsize].range = sqrt(pl[plsize].x * pl[plsize].x + pl[plsize].y * pl[plsize].y); 
      // 特征提取
      give_feature(pl, types);
      // pl_surf += pl;
    }
    time += omp_get_wtime() - t0;
    printf("Feature extraction time: %lf \n", time / count);
  }
  // Step 4.2.1 不提取特征点
  else
  {
    for(uint i=1; i<plsize; i++)
    {
      // 线数合法 && 回波次序为0或1的点云
      if((msg->points[i].line < N_SCANS) && ((msg->points[i].tag & 0x30) == 0x10 || (msg->points[i].tag & 0x30) == 0x00))
      {
        valid_num ++; // 有线点云数量记录
        // Step 4.2.2 等间隔降采样
        if (valid_num % point_filter_num == 0)
        {
          pl_full[i].x = msg->points[i].x;
          pl_full[i].y = msg->points[i].y;
          pl_full[i].z = msg->points[i].z;
          pl_full[i].intensity = msg->points[i].reflectivity;
          pl_full[i].curvature = msg->points[i].offset_time / float(1000000); // use curvature as time of each laser points, curvature unit: ms
          
          // Step 4.2.3 当前点和上一个点的距离 > 1e-7 并且大于最小有效距离，才存储该点
          // 直接将该点存储为面点pl_surf
          if((abs(pl_full[i].x - pl_full[i-1].x) > 1e-7) 
              || (abs(pl_full[i].y - pl_full[i-1].y) > 1e-7)
              || (abs(pl_full[i].z - pl_full[i-1].z) > 1e-7)
              && (pl_full[i].x * pl_full[i].x + pl_full[i].y * pl_full[i].y + pl_full[i].z * pl_full[i].z > (blind * blind)))
          {
            pl_surf.push_back(pl_full[i]);
          }
        }
      }
    }
  }
}

/// @brief 进行oust64雷达数据预处理
/// @param msg 输入数据
void Preprocess::oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  pl_surf.clear();
  pl_corn.clear();
  pl_full.clear();
  pcl::PointCloud<ouster_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  int plsize = pl_orig.size();
  pl_corn.reserve(plsize);
  pl_surf.reserve(plsize);

  if (feature_enabled)
  {
    for (int i = 0; i < N_SCANS; i++)
    {
      pl_buff[i].clear();
      pl_buff[i].reserve(plsize);
    }

    for (uint i = 0; i < plsize; i++)
    {
      double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y + pl_orig.points[i].z * pl_orig.points[i].z;
      if (range < (blind * blind)) continue;
      Eigen::Vector3d pt_vec;
      PointType added_pt;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      // 后续yaw_angle也没有用到
      double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.3; // 57.3 = 180/ 3.14
      if (yaw_angle >= 180.0)
        yaw_angle -= 360.0;
      if (yaw_angle <= -180.0)
        yaw_angle += 360.0;

      added_pt.curvature = pl_orig.points[i].t * time_unit_scale;
      if(pl_orig.points[i].ring < N_SCANS)
      {
        pl_buff[pl_orig.points[i].ring].push_back(added_pt);
      }
    }

    for (int j = 0; j < N_SCANS; j++)
    {
      PointCloudXYZI &pl = pl_buff[j];
      int linesize = pl.size();
      vector<orgtype> &types = typess[j];
      types.clear();
      types.resize(linesize);
      linesize--;
      for (uint i = 0; i < linesize; i++)
      {
        types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
        vx = pl[i].x - pl[i + 1].x;
        vy = pl[i].y - pl[i + 1].y;
        vz = pl[i].z - pl[i + 1].z;
        types[i].dista = vx * vx + vy * vy + vz * vz;
      }
      types[linesize].range = sqrt(pl[linesize].x * pl[linesize].x + pl[linesize].y * pl[linesize].y);
      give_feature(pl, types);
    }
  }
  else
  {
    double time_stamp = msg->header.stamp.toSec();
    // cout << "===================================" << endl;
    // printf("Pt size = %d, N_SCANS = %d\r\n", plsize, N_SCANS);
    for (int i = 0; i < pl_orig.points.size(); i++)
    {
      // 等间隔降采样
      if (i % point_filter_num != 0) continue;

      double range = pl_orig.points[i].x * pl_orig.points[i].x + pl_orig.points[i].y * pl_orig.points[i].y + pl_orig.points[i].z * pl_orig.points[i].z;
      
      if (range < (blind * blind)) continue;
      
      Eigen::Vector3d pt_vec;
      PointType added_pt;
      added_pt.x = pl_orig.points[i].x;
      added_pt.y = pl_orig.points[i].y;
      added_pt.z = pl_orig.points[i].z;
      added_pt.intensity = pl_orig.points[i].intensity;
      added_pt.normal_x = 0;
      added_pt.normal_y = 0;
      added_pt.normal_z = 0;
      added_pt.curvature = pl_orig.points[i].t * time_unit_scale; // curvature unit: ms

      pl_surf.points.push_back(added_pt);
    }
  }
  // pub_func(pl_surf, pub_full, msg->header.stamp);
  // pub_func(pl_surf, pub_corn, msg->header.stamp);
}

/// @brief velodyne雷达数据预处理
/// @param msg 输入数据
void Preprocess::velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    pl_surf.clear();
    pl_corn.clear();
    pl_full.clear();

    pcl::PointCloud<velodyne_ros::Point> pl_orig;
    pcl::fromROSMsg(*msg, pl_orig);
    int plsize = pl_orig.points.size();
    if (plsize == 0) return;
    pl_surf.reserve(plsize);

    /*** These variables only works when no point timestamps given ***/
    double omega_l = 0.361 * SCAN_RATE;       // scan angular velocity
    std::vector<bool> is_first(N_SCANS,true);
    std::vector<double> yaw_fp(N_SCANS, 0.0);      // yaw of first scan point
    std::vector<float> yaw_last(N_SCANS, 0.0);   // yaw of last scan point
    std::vector<float> time_last(N_SCANS, 0.0);  // last offset time
    /*****************************************************************/

    if (pl_orig.points[plsize - 1].time > 0)
    {
      given_offset_time = true;
    }
    else
    {
      given_offset_time = false;
      double yaw_first = atan2(pl_orig.points[0].y, pl_orig.points[0].x) * 57.29578;
      double yaw_end  = yaw_first;
      int layer_first = pl_orig.points[0].ring;
      for (uint i = plsize - 1; i > 0; i--)
      {
        if (pl_orig.points[i].ring == layer_first)
        {
          yaw_end = atan2(pl_orig.points[i].y, pl_orig.points[i].x) * 57.29578;
          break;
        }
      }
    }

    if(feature_enabled)
    {
      for (int i = 0; i < N_SCANS; i++)
      {
        pl_buff[i].clear();
        pl_buff[i].reserve(plsize);
      }
      
      for (int i = 0; i < plsize; i++)
      {
        PointType added_pt;
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        int layer  = pl_orig.points[i].ring;
        if (layer >= N_SCANS) continue;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = pl_orig.points[i].time * time_unit_scale; // units: ms

        if (!given_offset_time)
        {
          double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;
          if (is_first[layer]) // 是否在第一圈上
          {
            // printf("layer: %d; is first: %d", layer, is_first[layer]);
              yaw_fp[layer]=yaw_angle;
              is_first[layer]=false;
              added_pt.curvature = 0.0;
              yaw_last[layer]=yaw_angle;
              time_last[layer]=added_pt.curvature;
              continue;
          }

          if (yaw_angle <= yaw_fp[layer])
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle) / omega_l;
          }
          else
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle+360.0) / omega_l;
          }

          if (added_pt.curvature < time_last[layer])  added_pt.curvature+=360.0/omega_l;

          yaw_last[layer] = yaw_angle;
          time_last[layer]=added_pt.curvature;
        }

        pl_buff[layer].points.push_back(added_pt);
      }

      for (int j = 0; j < N_SCANS; j++)
      {
        PointCloudXYZI &pl = pl_buff[j];
        int linesize = pl.size();
        if (linesize < 2) continue;
        vector<orgtype> &types = typess[j];
        types.clear();
        types.resize(linesize);
        linesize--;
        for (uint i = 0; i < linesize; i++)
        {
          types[i].range = sqrt(pl[i].x * pl[i].x + pl[i].y * pl[i].y);
          vx = pl[i].x - pl[i + 1].x;
          vy = pl[i].y - pl[i + 1].y;
          vz = pl[i].z - pl[i + 1].z;
          types[i].dista = vx * vx + vy * vy + vz * vz;
        }
        types[linesize].range = sqrt(pl[linesize].x * pl[linesize].x + pl[linesize].y * pl[linesize].y);
        give_feature(pl, types);
      }
    }
    else
    {
      for (int i = 0; i < plsize; i++)
      {
        PointType added_pt;
        // cout<<"!!!!!!"<<i<<" "<<plsize<<endl;
        
        added_pt.normal_x = 0;
        added_pt.normal_y = 0;
        added_pt.normal_z = 0;
        added_pt.x = pl_orig.points[i].x;
        added_pt.y = pl_orig.points[i].y;
        added_pt.z = pl_orig.points[i].z;
        added_pt.intensity = pl_orig.points[i].intensity;
        added_pt.curvature = pl_orig.points[i].time * time_unit_scale;  // curvature unit: ms // cout<<added_pt.curvature<<endl;

        if (!given_offset_time)
        {
          int layer = pl_orig.points[i].ring;
          double yaw_angle = atan2(added_pt.y, added_pt.x) * 57.2957;

          if (is_first[layer])
          {
            // printf("layer: %d; is first: %d", layer, is_first[layer]);
              yaw_fp[layer]=yaw_angle;
              is_first[layer]=false;
              added_pt.curvature = 0.0;
              yaw_last[layer]=yaw_angle;
              time_last[layer]=added_pt.curvature;
              continue;
          }

          // compute offset time
          if (yaw_angle <= yaw_fp[layer])
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle) / omega_l;
          }
          else
          {
            added_pt.curvature = (yaw_fp[layer]-yaw_angle+360.0) / omega_l;
          }

          if (added_pt.curvature < time_last[layer])  added_pt.curvature+=360.0/omega_l;

          yaw_last[layer] = yaw_angle;
          time_last[layer]=added_pt.curvature;
        }

        // 等间隔降采样
        if (i % point_filter_num == 0)
        {
          if(added_pt.x*added_pt.x+added_pt.y*added_pt.y+added_pt.z*added_pt.z > (blind * blind))
          {
            pl_surf.points.push_back(added_pt);
          }
        }
      }
    }
}

/// @brief 提取特征点，边缘点和面点分别记录在pl_corn和pl_surfs中
/// @param pl in&out 输入点云 / 输出特征点
/// @param types in&out 输入包含了点云在xy平面上距离原点和下一个点的距离 / 输出更多信息
void Preprocess::give_feature(pcl::PointCloud<PointType> &pl, vector<orgtype> &types)
{
  int plsize = pl.size(); // 当前scan的点云数
  int plsize2;
  if(plsize == 0) // 没有点云数量，错误
  {
    printf("something wrong\n");
    return;
  }

  // head记录在当前scan中不在盲区中的第一个点的ID
  uint head = 0;
  while(types[head].range < blind)
  {
    head++;
  }

  // 判断是否为面点：Surf
  // group_size默认值是8，如果当前scan中的点云数量>group_size，那么遍历head到plsize - group_size的点
  // 否则plsize2 = 0，就不会进行遍历
  plsize2 = (plsize > group_size) ? (plsize - group_size) : 0;

  Eigen::Vector3d curr_direct(Eigen::Vector3d::Zero());
  Eigen::Vector3d last_direct(Eigen::Vector3d::Zero());

  uint i_nex = 0, i2; // TODO：i2???
  uint last_i = 0; uint last_i_nex = 0;
  int last_state = 0; // 记录上一点是否是面点，如果1，那么是；0，不是面点
  int plane_type; // 面点的类型

  // 遍历
  for(uint i=head; i<plsize2; i++)
  {
    if(types[i].range < blind) // 如果当前点xy平面上的距离小于盲区，跳过该点
    {
      continue;
    }

    i2 = i;

    // 判断当前点是否为面点
    // plane_type = 0 1 2
    plane_type = plane_judge(pl, types, i, i_nex, curr_direct);
    
    // 如果plane_type==1 ，说明该点有可能是面点
    if(plane_type == 1)
    {
      // 遍历 i ~ i_nex，如果不是起始点和终点，认为是面点；其余认为是可能的面点
      for(uint j=i; j<=i_nex; j++)
      { 
        if(j!=i && j!=i_nex)
        {
          types[j].ftype = Real_Plane;
        }
        else
        {
          types[j].ftype = Poss_Plane;
        }
      }
      
      // if(last_state==1 && fabs(last_direct.sum())>0.5)
      // 判断该点为两平面边上的面点还是较平坦的面点
      // 如果上一个点是面点，且其法向量长度>0.1
      if(last_state==1 && last_direct.norm()>0.1)
      {
        // 因为法向量都是归一化处理了的，所以这里mod直接就是sin值
        double mod = last_direct.transpose() * curr_direct;
        // 如果两法向量的夹角是45～135度之间，认为当前点是边上的面点；否则认为是较平坦的面点
        if(mod>-0.707 && mod<0.707)
        {
          types[i].ftype = Edge_Plane;
        }
        else
        {
          types[i].ftype = Real_Plane;
        }
      }
      
      i = i_nex - 1; // 记录
      last_state = 1;
    }
    else // if(plane_type == 2)
    {
      i = i_nex;
      last_state = 0; // 不是面点
    }
    // else if(plane_type == 0)
    // {
    //   if(last_state == 1)
    //   {
    //     uint i_nex_tem;
    //     uint j;
    //     for(j=last_i+1; j<=last_i_nex; j++)
    //     {
    //       uint i_nex_tem2 = i_nex_tem;
    //       Eigen::Vector3d curr_direct2;

    //       uint ttem = plane_judge(pl, types, j, i_nex_tem, curr_direct2);

    //       if(ttem != 1)
    //       {
    //         i_nex_tem = i_nex_tem2;
    //         break;
    //       }
    //       curr_direct = curr_direct2;
    //     }

    //     if(j == last_i+1)
    //     {
    //       last_state = 0;
    //     }
    //     else
    //     {
    //       for(uint k=last_i_nex; k<=i_nex_tem; k++)
    //       {
    //         if(k != i_nex_tem)
    //         {
    //           types[k].ftype = Real_Plane;
    //         }
    //         else
    //         {
    //           types[k].ftype = Poss_Plane;
    //         }
    //       }
    //       i = i_nex_tem-1;
    //       i_nex = i_nex_tem;
    //       i2 = j-1;
    //       last_state = 1;
    //     }

    //   }
    // }

    last_i = i2; // 记录上一个点的索引
    last_i_nex = i_nex; // 记录上一次用于判断面点的点云集合中最后一个点的索引(感觉也没啥用)
    last_direct = curr_direct; // 记录上一个面的法向量
  }

  // 判断是否是边缘点
  plsize2 = plsize > 3 ? plsize - 3 : 0;
  for(uint i=head+3; i<plsize2; i++)
  {
    // 如果该点在盲区内，或者已经判定是确定的面点，直接跳过该点的遍历
    if(types[i].range<blind || types[i].ftype>=Real_Plane)
    {
      continue;
    }

    // 该点与前后点的距离太近，跳过
    if(types[i-1].dista<1e-16 || types[i].dista<1e-16)
    {
      continue;
    }

    Eigen::Vector3d vec_a(pl[i].x, pl[i].y, pl[i].z); // 该点在当前Lidar系下的3D位置
    Eigen::Vector3d vecs[2];

    // 两次遍历
    for(int j=0; j<2; j++)
    {
      int m = -1;
      if(j == 1)
      {
        m = 1;
      }

      // m=-1或m=1，意味这判断点的前后两点是否在盲区内
      if(types[i+m].range < blind)
      {
        // 如果点的前后两点在盲区内，而且当前点的range大于10m，认为该点跳变比较大
        if(types[i].range > inf_bound) // inf_bound = 10m
        {
          types[i].edj[j] = Nr_inf;
        }
        else
        {
          types[i].edj[j] = Nr_blind;
        }
        continue;
      }

      // 设激光原点为点O，当前点为点A，i+m点为点B
      vecs[j] = Eigen::Vector3d(pl[i+m].x, pl[i+m].y, pl[i+m].z);
      vecs[j] = vecs[j] - vec_a; // AB
      types[i].angle[j] = vec_a.dot(vecs[j]) / vec_a.norm() / vecs[j].norm(); // cosOAB
      // cosOAB < cos(170)，意味 170 < OAB < 180，  OAB三点几乎在同一条直线上 ——即B在OA的延长线上
      if(types[i].angle[j] < jump_up_limit) // jump_up_limit = cos(170)
      {
        types[i].edj[j] = Nr_180;
      }
      // cosOAB > cos(8)，意味 0 < OAB < 8，  OAB三点几乎在同一条直线上 —— 但B在OA线段上
      else if(types[i].angle[j] > jump_down_limit) // jump_down_limit = cos(8)
      {
        types[i].edj[j] = Nr_zero;
      }
    }

    // 以下的判定大同小异，都是先判断该点不平行于激光束，然后判断该点是否可能被遮挡，和LOAM中边缘点的判断类似
    // vecs计算完毕，Prev = 0，Next = 1
    // 设激光原点为点O，当前点为点A，前一个点为M，后一个点为N
    // 计算cos(MANs)s
    types[i].intersect = vecs[Prev].dot(vecs[Next]) / vecs[Prev].norm() / vecs[Next].norm(); 
    // 如果前一个点是正常点且在OA延长线上 && 后一个点是正常点且在OA，即激光线上 && 当前点和后一个点的距离大于0.0225  && 当前点和前一个点的距离 大于 当前点和后一个点的距离的4倍
    if(types[i].edj[Prev]==Nr_nor && types[i].edj[Next]==Nr_zero && types[i].dista>0.0225 && types[i].dista>4*types[i-1].dista)
    {
      // 0 < MAN < 160，不能平行于激光线
      if(types[i].intersect > cos160)
      {
        if(edge_jump_judge(pl, types, i, Prev))
        {
          types[i].ftype = Edge_Jump;
        }
      }
    }
    else if(types[i].edj[Prev]==Nr_zero && types[i].edj[Next]== Nr_nor && types[i-1].dista>0.0225 && types[i-1].dista>4*types[i].dista)
    {
      if(types[i].intersect > cos160)
      {
        if(edge_jump_judge(pl, types, i, Next))
        {
          types[i].ftype = Edge_Jump;
        }
      }
    }
    else if(types[i].edj[Prev]==Nr_nor && types[i].edj[Next]==Nr_inf)
    {
      if(edge_jump_judge(pl, types, i, Prev))
      {
        types[i].ftype = Edge_Jump;
      }
    }
    else if(types[i].edj[Prev]==Nr_inf && types[i].edj[Next]==Nr_nor)
    {
      if(edge_jump_judge(pl, types, i, Next))
      {
        types[i].ftype = Edge_Jump;
      }
     
    }
    else if(types[i].edj[Prev]>Nr_nor && types[i].edj[Next]>Nr_nor)
    {
      if(types[i].ftype == Nor)
      {
        types[i].ftype = Wire;
      }
    }
  }

  // 继续寻找面点
  // 这个和LOAM中边缘点和面点的寻找很类似：找到边缘点和面点后，剩下的点都认为是面点，是符合实际情况的
  plsize2 = plsize-1;
  double ratio;
  for(uint i=head+1; i<plsize2; i++)
  {
    // 当前点和前后两点都不在盲区，才继续
    if(types[i].range<blind || types[i-1].range<blind || types[i+1].range<blind)
    {
      continue;
    }
    
    // 当前点和前后两点不能太近
    if(types[i-1].dista<1e-8 || types[i].dista<1e-8)
    {
      continue;
    }

    // 如果当前点是正常点
    if(types[i].ftype == Nor)
    {
      // 计算当前点到上一个点的距离 和 当前点到下一个点的距离 的比例
      if(types[i-1].dista > types[i].dista)
      {
        ratio = types[i-1].dista / types[i].dista;
      }
      else
      {
        ratio = types[i].dista / types[i-1].dista;
      }

      // smallp_intersect = cos(172.5) , smallp_ratio = 1.2
      // 如果 172.5 < MAN < 180，间距比例 < 1.2；且前后点都是正常点的情况下，都认为是面点
      if(types[i].intersect<smallp_intersect && ratio < smallp_ratio)
      {
        if(types[i-1].ftype == Nor)
        {
          types[i-1].ftype = Real_Plane;
        }
        if(types[i+1].ftype == Nor)
        {
          types[i+1].ftype = Real_Plane;
        }
        types[i].ftype = Real_Plane;
      }
    }
  }

  // 存储边缘点和面点
  int last_surface = -1;
  for(uint j=head; j<plsize; j++)
  {
    if(types[j].ftype==Poss_Plane || types[j].ftype==Real_Plane)
    {
      if(last_surface == -1)
      {
        last_surface = j;
      }

      // 面点的数量是很多的，所以这里每隔几个点才存储一个面点，达到滤波的目的
      if(j == uint(last_surface+point_filter_num-1))
      {
        PointType ap;
        ap.x = pl[j].x;
        ap.y = pl[j].y;
        ap.z = pl[j].z;
        ap.intensity = pl[j].intensity;
        ap.curvature = pl[j].curvature;
        pl_surf.push_back(ap);

        last_surface = -1;
      }
    }
    else
    {
      if(types[j].ftype==Edge_Jump || types[j].ftype==Edge_Plane)
      {
        pl_corn.push_back(pl[j]);
      }
      if(last_surface != -1)
      {
        // 计算上次面点和这次边缘点之间点的中心点ap，将该点当成面点进行存储
        PointType ap;
        for(uint k=last_surface; k<j; k++)
        {
          ap.x += pl[k].x;
          ap.y += pl[k].y;
          ap.z += pl[k].z;
          ap.intensity += pl[k].intensity;
          ap.curvature += pl[k].curvature;
        }
        ap.x /= (j-last_surface);
        ap.y /= (j-last_surface);
        ap.z /= (j-last_surface);
        ap.intensity /= (j-last_surface);
        ap.curvature /= (j-last_surface);
        pl_surf.push_back(ap);
      }
      last_surface = -1;
    }
  }
}

void Preprocess::pub_func(PointCloudXYZI &pl, const ros::Time &ct)
{
  pl.height = 1; pl.width = pl.size();
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(pl, output);
  output.header.frame_id = "livox";
  output.header.stamp = ct;
}

/// @brief 判断当前点的类型
/// @param[in out] pl 当前scan上的所有点云集合
/// @param[in out] types 当前scan上的所有点云信息集合
/// @param[in] i_cur 当前点云在pl上的ID
/// @param[out] i_nex  用于拟合平面的最后一个点的索引值
/// @param[out] curr_direct 法向量
/// @return 
int Preprocess::plane_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct)
{
  // disA = 0.01, disB = 0.1
  // group_dis = 0.01*(range) + 0.1
  double group_dis = disA*types[i_cur].range + disB;
  group_dis = group_dis * group_dis;
  // i_nex = i_cur;

  double two_dis;
  vector<double> disarr;
  disarr.reserve(20);

  // group_size = 8
  // 遍历当前点 ～ 当前点+group_size共八个点
  for(i_nex=i_cur; i_nex<i_cur+group_size; i_nex++)
  {
    // 如果遍历过程中有一个点的range在盲区内，那么直接返回2，注意这里的2不是指该点被却认为平面点了
    // 因为这个点离原点太近了
    if(types[i_nex].range < blind)
    {
      curr_direct.setZero(); // 距离原点太小了，将法向量设置为0
      return 2;
    }
    disarr.push_back(types[i_nex].dista); // 存储当前索引为i_nex的点云和其下一个点的距离
  }
  
  // 查看后续点有没有可以使用的
  for(;;)
  {
    // 如果当前点和当前点后i_nex个点已经超出pl点云集合范围，退出遍历
    if((i_cur >= pl.size()) || (i_nex >= pl.size())) break;

    // 再次判断点云是否距离原点太近
    if(types[i_nex].range < blind)
    {
      curr_direct.setZero();
      return 2;
    }
    vx = pl[i_nex].x - pl[i_cur].x;
    vy = pl[i_nex].y - pl[i_cur].y;
    vz = pl[i_nex].z - pl[i_cur].z;
    two_dis = vx*vx + vy*vy + vz*vz;
    // TODO：有疑问就是，pl中后续点并不是按照某种距离排序的，而是根据扫描时间依次获得
    // 为什么仅当出现一次(two_dis >= group_dis)就结束整个遍历呢？
    // 我猜：或许是因为本来这个for的目的就是为了找出额外可利用的点，但并不是一定要找到！本身八个点也是够用的
    // 所以为了时间和精度的话，做了折中？？？
    if(two_dis >= group_dis) // 距离当前点太远，直接退出搜索
    {
      break;
    }
    disarr.push_back(types[i_nex].dista);
    i_nex++;
  }

  double leng_wid = 0;
  double v1[3], v2[3];
  // 设点i_cur 为 点A，点j为点B，点i_nex为点C
  // 上述已经能得到AC之间的距离的平方 = two_dis = |AC||AC|，在xyz三个方向上的距离差 AC =（vx, vy, vx）
  // 筛选出在i_cur和i_nex之间离AC距离最近的点
  for(uint j=i_cur+1; j<i_nex; j++)
  {
    if((j >= pl.size()) || (i_cur >= pl.size())) break;
    // 求AB
    v1[0] = pl[j].x - pl[i_cur].x;
    v1[1] = pl[j].y - pl[i_cur].y;
    v1[2] = pl[j].z - pl[i_cur].z;

    // 求AC X AB（AC叉乘AB）= |AC||AB|sin<AC，AB> ，其中|AB|sin<AC，AB> = h表示点B到AC的距离
    v2[0] = v1[1]*vz - vy*v1[2];
    v2[1] = v1[2]*vx - v1[0]*vz;
    v2[2] = v1[0]*vy - vx*v1[1];

    // 由A B C组成的平行四边形的面积的平方 lw = (|AC||AC|*h*h)
    double lw = v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2];
    if(lw > leng_wid) // 找到面积最大的，对应着离AC最远的点B
    {
      leng_wid = lw; 
    }
  }

  // 因为two_dis = |AC||AC|，leng_wid = |AC||AC|*h*h
  // 所以 two_dis*two_dis/leng_wid = 1 / (h*h) < 225
  // 结果就是 h > 1/15 = 0.067m
  // 这里的意思是不是要求B距离AC的距离要大于0.067米，否则点与点之间太近了，拟合的平面质量不太好
  // TODO：那 “(two_dis*two_dis/leng_wid) < p2l_ratio”应该改为 “(two_dis*two_dis/leng_wid) > p2l_ratio”，表示如果h < 0.067，那么该点就是一个正常点，不再进行平面的拟合
  // 但是FAST-LIO默认不开特征提取，所以估计对结果不会有什么影响
  if((two_dis*two_dis/leng_wid) < p2l_ratio) // p2l_ratio = 225
  {
    curr_direct.setZero(); // 太近了，所以法向量设置为0
    return 0;
  }

  // 将disarr按照由大到小进行排序
  uint disarrsize = disarr.size();
  for(uint j=0; j<disarrsize-1; j++)
  {
    for(uint k=j+1; k<disarrsize; k++)
    {
      if(disarr[j] < disarr[k])
      {
        leng_wid = disarr[j];
        disarr[j] = disarr[k];
        disarr[k] = leng_wid;
      }
    }
  }

  // 后两个点距离太近了，也是直接退出
  if(disarr[disarr.size()-2] < 1e-16)
  {
    curr_direct.setZero();
    return 0;
  }

  // 根据不同的雷达类型选择不同的判断依据
  if(lidar_type==AVIA)
  {
    double dismax_mid = disarr[0]/disarr[disarrsize/2];
    double dismid_min = disarr[disarrsize/2]/disarr[disarrsize-2];

    // limit_maxmid = limit_midmin = 6.25
    // 点和点之间的差距太大了，拟合的平面质量或许不太好
    if(dismax_mid>=limit_maxmid || dismid_min>=limit_midmin)
    {
      curr_direct.setZero();
      return 0;
    }
  }
  else
  {
    double dismax_min = disarr[0] / disarr[disarrsize-2];
    if(dismax_min >= limit_maxmin)
    {
      curr_direct.setZero();
      return 0;
    }
  }
  
  // AC为其法向量
  curr_direct << vx, vy, vz;
  curr_direct.normalize(); // 归一化
  return 1;
}

/// @brief 判断是否为边缘点
/// @param pl 当前点云集合
/// @param types 当前点云信息集合
/// @param i 当前遍历点云的索引值四
/// @param nor_dir 0 ：判断上一个点和当前点的关系；1：判断下一个点和当前点的关系
/// @return 
bool Preprocess::edge_jump_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, Surround nor_dir)
{
  // 判断上一个点和当前点的关系
  if(nor_dir == 0)
  {
    // 上一个点和上上个点都在盲区内，直接false，当前点不是边缘点
    if(types[i-1].range<blind || types[i-2].range<blind)
    {
      return false;
    }
  }
  // 判断下一个点和当前点的关系
  else if(nor_dir == 1)
  {
    // 下一个点和下下个点都在盲区内，直接false，当前点不是边缘点
    if(types[i+1].range<blind || types[i+2].range<blind)
    {
      return false;
    }
  }

  // nor_dir = 0时，i+nor_dir-1 = i-1，i+3*nor_dir-2 = i-2
  // nor_dir = 1时，i+nor_dir-1 = i，i+3*nor_dir-2 = i+1
  double d1 = types[i+nor_dir-1].dista;
  double d2 = types[i+3*nor_dir-2].dista;
  double d;

  // 排序后，d1 > d2
  if(d1<d2)
  {
    d = d1;
    d1 = d2;
    d2 = d;
  }

  d1 = sqrt(d1);
  d2 = sqrt(d2);

  // edgea = 2，edgeb = 0.1
  // 说明间格局离太大，有可能被遮挡，直接false，当前点不是边缘点
  if(d1>edgea*d2 || (d1-d2)>edgeb)
  {
    return false;
  }
  
  return true;
}
