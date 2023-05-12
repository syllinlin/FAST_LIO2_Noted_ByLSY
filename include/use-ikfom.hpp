#ifndef USE_IKFOM_H
#define USE_IKFOM_H

#include <IKFoM_toolkit/esekfom/esekfom.hpp>

typedef MTK::vect<3, double> vect3;
typedef MTK::SO3<double> SO3;
typedef MTK::S2<double, 98090, 10000, 1> S2; 
typedef MTK::vect<1, double> vect1;
typedef MTK::vect<2, double> vect2;

// 系统状态(在流形空间上定义)
MTK_BUILD_MANIFOLD(state_ikfom,
((vect3, pos)) // 位置
((SO3, rot))	 // 旋转
((SO3, offset_R_L_I)) // 外参：雷达相对于IMU的旋转
((vect3, offset_T_L_I)) // 外参：雷达相对于IMU的平移
((vect3, vel)) // 速度
((vect3, bg)) // 陀螺仪的测量偏置
((vect3, ba)) // 加速度计的测量偏置
((S2, grav)) // 重力加速度(这里定义在SO2上，因为重力矢量的方向是确定的，所以减少一维，定义在SO2上)
);

// 输入：指IMU的加速度和角速度测量值
MTK_BUILD_MANIFOLD(input_ikfom,
((vect3, acc))
((vect3, gyro))
);

// IMU的测量噪声：包括陀螺仪测量噪声、加速度计测量噪声、陀螺仪测量偏置的倒数、加速度计测量偏置的倒数
MTK_BUILD_MANIFOLD(process_noise_ikfom,
((vect3, ng))
((vect3, na))
((vect3, nbg))
((vect3, nba))
);

/// @brief 完成噪声协方差初始化，包括加速度计测量噪声、陀螺仪测量噪声、加速度计偏置噪声、陀螺仪偏置噪声的协方差
/// @return 
MTK::get_cov<process_noise_ikfom>::type process_noise_cov()
{
	MTK::get_cov<process_noise_ikfom>::type cov = MTK::get_cov<process_noise_ikfom>::type::Zero();
	MTK::setDiagonal<process_noise_ikfom, vect3, 0>(cov, &process_noise_ikfom::ng, 0.0001);// 0.03
	MTK::setDiagonal<process_noise_ikfom, vect3, 3>(cov, &process_noise_ikfom::na, 0.0001); // *dt 0.01 0.01 * dt * dt 0.05
	MTK::setDiagonal<process_noise_ikfom, vect3, 6>(cov, &process_noise_ikfom::nbg, 0.00001); // *dt 0.00001 0.00001 * dt *dt 0.3 //0.001 0.0001 0.01
	MTK::setDiagonal<process_noise_ikfom, vect3, 9>(cov, &process_noise_ikfom::nba, 0.00001);   //0.001 0.05 0.0001/out 0.01
	return cov;
}

//double L_offset_to_I[3] = {0.04165, 0.02326, -0.0284}; // Avia 
//vect3 Lidar_offset_to_IMU(L_offset_to_I, 3);
/// @brief 完成FAST-LIO2中公式(5)中函数f的计算：f的作用就是计算两个IMU时刻之间状态的变化量
/// @param[in] s 上一时刻的系统状态量
/// @param[in] in 输入的IMU测量值 
/// @return 状态更新函数f，维度为24x1
Eigen::Matrix<double, 24, 1> get_f(state_ikfom &s, const input_ikfom &in)
{
	// 注意，f位置和论文里的定义不一致
	// 系统状态量顺序分别是 —— [p R R_IL P_IL V bg ba g]
	// 24维依次对应速度(3)，角速度(3)，外参偏置T(3)=0,，外参偏置R(3)=0，加速度(3)，角速度偏置(3)，加速度偏置(3)，重力矢量更新(3)
	Eigen::Matrix<double, 24, 1> res = Eigen::Matrix<double, 24, 1>::Zero(); // 初始化为0
	vect3 omega;
	in.gyro.boxminus(omega, s.bg); // omega = in.gyro - s.bg = 测量值- 偏置
	vect3 a_inertial = s.rot * (in.acc-s.ba); // (加速度测量值-偏置)后转到Global系下(世界坐标系)
	for(int i = 0; i < 3; i++ )
	{
		res(i) = s.vel[i]; //更新速度 —— 没有考虑加速度，因该将两个时刻之间的运动作为匀速运动
		res(i + 3) =  omega[i]; //更新角速度 —— 对应 R 
		res(i + 12) = a_inertial[i] + s.grav[i]; //更新加速度 —— 对应 V
	}
	return res;
}

/// @brief 状态的误差值相对于误差状态的偏导
/// @param s 系统状态
/// @param in 测量值
/// @return F_dx(用dx表示x_real - x_mea，即状态误差)，维度是24x23
Eigen::Matrix<double, 24, 23> df_dx(state_ikfom &s, const input_ikfom &in)
{
	// 当中的23个对应了status的维度计算，为pos(3), rot(3)，offset_R_L_I(3)，offset_T_L_I(3), vel(3)，bg(3)，ba(3)，grav(2)
	// status和f对应位置的物理含义都和state_ikfom定义顺序一致
	Eigen::Matrix<double, 24, 23> cov = Eigen::Matrix<double, 24, 23>::Zero(); // 初始化为0
	cov.template block<3, 3>(0, 12) = Eigen::Matrix3d::Identity();
	vect3 acc_;
	in.acc.boxminus(acc_, s.ba);
	vect3 omega;
	in.gyro.boxminus(omega, s.bg);
	cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix()*MTK::hat(acc_);
	cov.template block<3, 3>(12, 18) = -s.rot.toRotationMatrix();
	Eigen::Matrix<state_ikfom::scalar, 2, 1> vec = Eigen::Matrix<state_ikfom::scalar, 2, 1>::Zero();
	Eigen::Matrix<state_ikfom::scalar, 3, 2> grav_matrix;
	s.S2_Mx(grav_matrix, vec, 21);
	cov.template block<3, 2>(12, 21) =  grav_matrix; 
	cov.template block<3, 3>(3, 15) = -Eigen::Matrix3d::Identity(); 
	return cov;
}

/// @brief 状态的误差值相对于噪声的偏导
/// @param s 系统状态
/// @param in 测量值
/// @return F_w(用w表示噪声)，维度是24x12
Eigen::Matrix<double, 24, 12> df_dw(state_ikfom &s, const input_ikfom &in)
{
	Eigen::Matrix<double, 24, 12> cov = Eigen::Matrix<double, 24, 12>::Zero();
	cov.template block<3, 3>(12, 3) = -s.rot.toRotationMatrix(); // 速度状态的误差对加速度噪声的偏导数
	cov.template block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity(); // 旋转状态的误差对角速度噪声的偏导数
	cov.template block<3, 3>(15, 6) = Eigen::Matrix3d::Identity(); // 角速度偏置对角速度偏置噪声的偏导数
	cov.template block<3, 3>(18, 9) = Eigen::Matrix3d::Identity(); // 加速度偏置对加速度偏置噪声的偏导数
	return cov;
}

/// @brief 从SO3流形空间转到欧拉角表达
/// @param orient 
/// @return 
vect3 SO3ToEuler(const SO3 &orient) 
{
	Eigen::Matrix<double, 3, 1> _ang;
	Eigen::Vector4d q_data = orient.coeffs().transpose();
	//scalar w=orient.coeffs[3], x=orient.coeffs[0], y=orient.coeffs[1], z=orient.coeffs[2];
	double sqw = q_data[3]*q_data[3];
	double sqx = q_data[0]*q_data[0];
	double sqy = q_data[1]*q_data[1];
	double sqz = q_data[2]*q_data[2];
	double unit = sqx + sqy + sqz + sqw; // if normalized is one, otherwise is correction factor
	double test = q_data[3]*q_data[1] - q_data[2]*q_data[0];

	if (test > 0.49999*unit) { // singularity at north pole
	
		_ang << 2 * std::atan2(q_data[0], q_data[3]), M_PI/2, 0;
		double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
		vect3 euler_ang(temp, 3);
		return euler_ang;
	}
	if (test < -0.49999*unit) { // singularity at south pole
		_ang << -2 * std::atan2(q_data[0], q_data[3]), -M_PI/2, 0;
		double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
		vect3 euler_ang(temp, 3);
		return euler_ang;
	}
		
	_ang <<
			std::atan2(2*q_data[0]*q_data[3]+2*q_data[1]*q_data[2] , -sqx - sqy + sqz + sqw),
			std::asin (2*test/unit),
			std::atan2(2*q_data[2]*q_data[3]+2*q_data[1]*q_data[0] , sqx - sqy - sqz + sqw);
	double temp[3] = {_ang[0] * 57.3, _ang[1] * 57.3, _ang[2] * 57.3};
	vect3 euler_ang(temp, 3);
		// euler_ang[0] = roll, euler_ang[1] = pitch, euler_ang[2] = yaw
	return euler_ang;
}

#endif