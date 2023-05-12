#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <livox_ros_driver/CustomMsg.h>

using namespace std;

#define IS_VALID(a)  ((abs(a)>1e8) ? true : false)

typedef pcl::PointXYZINormal PointType;
typedef pcl::PointCloud<PointType> PointCloudXYZI;

// 枚举类型：表示支持的雷达类型
enum LID_TYPE
{
  AVIA = 1, 
  VELO16, 
  OUST64
}; //{1, 2, 3}

// 时间单位
enum TIME_UNIT
{
  SEC = 0,  // 秒
  MS = 1,    // 毫秒
  US = 2,     // 微秒
  NS = 3      // 纳秒
};

// 特征点状态，或称为特征点类型
enum Feature
{
  Nor,                  // normal 正常点
  Poss_Plane,  // possible plane feature 可能的平面点
  Real_Plane,   // real plane feature 确定的平面点
  Edge_Jump, // 有跨越的边 TODO：？？？—— 指的应该是“边缘点”
  Edge_Plane, // 平面边缘的点——边缘点的一种
  Wire,                // 线段 TODO
  ZeroPoint      // 无效点（在程序中并没有使用）
};

// 某点云的位置标记
enum Surround
{
  Prev, // 在当前点云之前
  Next  // 在当前点云之后
};

// 有跨越的边的状态
// TODO：每一个都表示什么含义？？？
enum E_jump
{
  Nr_nor,   // 正常
  Nr_zero, // 0 —— 该点A的前后点在OA线上（O是对应Lidar系的原点）
  Nr_180,   // 180 —— 该点A的前后点在OA延长线上（O是对应Lidar系的原点）
  Nr_inf,     // 无穷——该点和周围的点的range相比很大，认为跳变比较大
  Nr_blind // 盲区
};

// 用于存储激光雷达点的一些额外的属性
struct orgtype
{
  double range;       // 该点云在xy平面距离雷达中心的位置
  double dista;         // 当前点与后一个点的距离 TODO：这个距离应该指得是在当前Lidar系下的绝对欧式距离
  // 即设当前Lidar系的原点为O，当前点云A的前一个点为M ，后一个点为N
  double angle[2];  // 分别存储角OAM和角OAN的cos值
  double intersect; // 存储角MAN的cos值
  E_jump edj[2];      // 前后两点的类型——判断前后两点和线段OA的位置关系：在OA线段上或者在OA的延长线上
  Feature ftype;       // 当前点云的类型
  // 初始化
  orgtype()
  {
    range = 0;
    edj[Prev] = Nr_nor; // 初始化的时候点的状态都是正常的
    edj[Next] = Nr_nor;
    ftype = Nor;
    intersect = 2;
  }
};

// 定义不同的雷达数据类型，根据实际使用雷达类型选择数据结构
namespace velodyne_ros // velodyne雷达
{
  struct EIGEN_ALIGN16 Point // 雷达点云数据结构
  {
      PCL_ADD_POINT4D;  // (有 x、y、z 还有一个对齐变量）添加点云的xyz坐标
      float intensity;              // 强度
      float time;                       // 时间
      uint16_t ring;                // 点所属圈数
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
}  // namespace velodyne_ros
// 注册点云信息
POINT_CLOUD_REGISTER_POINT_STRUCT(velodyne_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    (float, time, time)
    (uint16_t, ring, ring)
)

// ouster雷达点云数据结构
namespace ouster_ros 
{
  struct EIGEN_ALIGN16 Point 
  {
      PCL_ADD_POINT4D; 
      float intensity;              // (有 x、y、z 还有一个对齐变量）添加点云的xyz坐标
      uint32_t t;                      // 时间
      uint16_t reflectivity;  // 反射率
      uint8_t  ring;                  // 点所属圈数
      uint16_t ambient;       // TODO
      uint32_t range;             // 距离，这个应该是在当前Lidar系下（xyz平面）的3D距离
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW // 内存对齐
  };
}  // namespace ouster_ros
// 点云类型注册
// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(ouster_ros::Point,
    (float, x, x)
    (float, y, y)
    (float, z, z)
    (float, intensity, intensity)
    // use std::uint32_t to avoid conflicting with pcl::uint32_t
    (std::uint32_t, t, t)
    (std::uint16_t, reflectivity, reflectivity)
    (std::uint8_t, ring, ring)
    (std::uint16_t, ambient, ambient)
    (std::uint32_t, range, range)
)

class Preprocess
{
  public:
  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Preprocess();
  ~Preprocess();
  
  // 主要用于对Livox自定义Msg格式的激光雷达数据预处理
  void process(const livox_ros_driver::CustomMsg::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);
  // 其它激光雷达数据都是采用ros自带的Msg格式，所以用这个函数进行数据预处理
  void process(const sensor_msgs::PointCloud2::ConstPtr &msg, PointCloudXYZI::Ptr &pcl_out);
  void set(bool feat_en, int lid_type, double bld, int pfilt_num);

  // sensor_msgs::PointCloud2::ConstPtr pointcloud;
  PointCloudXYZI pl_full, pl_corn, pl_surf; // 全部点，边缘点，平面点
  PointCloudXYZI pl_buff[128]; //maximum 128 line lidar
  vector<orgtype> typess[128]; //maximum 128 line lidar
  float time_unit_scale; // TODO
  int lidar_type, point_filter_num, N_SCANS, SCAN_RATE, time_unit; // 雷达类型，采样间隔时间，扫描线数，扫描频率，时间单位
  double blind; // 最小有效距离，也就是盲区范围
  bool feature_enabled, given_offset_time; // 标志位：是否进行特征提取，是否进行时间补偿（偏移）
  ros::Publisher pub_full, pub_surf, pub_corn; // 发布全部点，发布面点和发布边缘点
    
  private:
  void avia_handler(const livox_ros_driver::CustomMsg::ConstPtr &msg);
  void oust64_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void velodyne_handler(const sensor_msgs::PointCloud2::ConstPtr &msg);
  void give_feature(PointCloudXYZI &pl, vector<orgtype> &types);
  void pub_func(PointCloudXYZI &pl, const ros::Time &ct);
  int  plane_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, uint &i_nex, Eigen::Vector3d &curr_direct);
  bool small_plane(const PointCloudXYZI &pl, vector<orgtype> &types, uint i_cur, uint &i_nex, Eigen::Vector3d &curr_direct);
  bool edge_jump_judge(const PointCloudXYZI &pl, vector<orgtype> &types, uint i, Surround nor_dir);
  
  int group_size;
  double disA, disB, inf_bound;
  double limit_maxmid, limit_midmin, limit_maxmin;
  double p2l_ratio;
  double jump_up_limit, jump_down_limit;
  double cos160;
  double edgea, edgeb;
  double smallp_intersect, smallp_ratio;
  double vx, vy, vz;
};
