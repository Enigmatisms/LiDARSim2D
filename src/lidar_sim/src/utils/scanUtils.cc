#include <random>
#include "utils/scanUtils.hpp"

double K_P = 0.2;
double K_I = 0.0001;
double K_D = 0.001;

void initializeReader(std::ifstream& recorded_path, std::vector<std::array<double, 8>>& path_vec, Eigen::Vector2d& init_obs, double& init_angle) {
    std::string line;
    getline(recorded_path, line);
    if (line != "version: 0.0.1")
        throw std::runtime_error("Recorded control input version incompatible.\n");
    getline(recorded_path, line);
    std::stringstream ss;
    ss << line;
    ss >> init_obs(0) >> init_obs(1) >> init_angle;
    while (getline(recorded_path, line)) {
        std::stringstream ss;
        ss << line;
        std::array<double, 8> pos;
        ss >> pos[0] >> pos[1] >> pos[2] >> pos[3] >> pos[4] >> pos[5] >> pos[6] >> pos[7];
        path_vec.push_back(pos);
    }
    recorded_path.close();
}

void initializeWriter(std::ofstream& record_output, const Eigen::Vector2d& init_obs, double init_angle) {
    record_output << "version: 0.0.1" << std::endl;
    record_output << init_obs.x() << " "  << init_obs.y() << " " << init_angle << std::endl;
}

void stampedTransform2TFMsg(const tf::StampedTransform& transf, tf::tfMessage& msg) {
    geometry_msgs::TransformStamped geo_msg;
    tf::transformStampedTFToMsg(transf, geo_msg);
    msg.transforms.clear();
    msg.transforms.push_back(geo_msg);
}

void makeTransform(Eigen::Vector3d p, std::string frame_id, std::string child_frame_id, tf::StampedTransform& transf) {
    tf::Matrix3x3 tf_R(cos(p.z()), -sin(p.z()), 0,
                       sin(p.z()), cos(p.z()), 0,
                       0, 0, 1);
    tf::Vector3 tf_t(p.x(), p.y(), 0);
    tf::Transform transform(tf_R, tf_t);
    transf = tf::StampedTransform(transform, ros::Time::now(), frame_id, child_frame_id);
}

tf::StampedTransform getOdom2MapTF(const tf::StampedTransform& scan2map, const tf::StampedTransform& scan2odom, const Eigen::Vector2d& init_obs, double init_angle) {
    double scan_rot = goodAngle(quaterion2Angle(scan2map.getRotation()) - init_angle), odom_rot = quaterion2Angle(scan2odom.getRotation()), inv_angle = goodAngle(scan_rot - odom_rot);
    Eigen::Matrix2d R_odom_inv;
    R_odom_inv << cos(odom_rot), sin(odom_rot), -sin(odom_rot), cos(odom_rot);
    const tf::Vector3 raw_scan_t = scan2map.getOrigin(), raw_odom_t = scan2odom.getOrigin();
    Eigen::Vector2d relative_t = Eigen::Vector2d(raw_scan_t.x(), raw_scan_t.y()) - init_obs;

    const Eigen::Vector2d odom_t(raw_odom_t.x(), raw_odom_t.y());
    const Eigen::Vector2d inv_t = relative_t - odom_t;
    // scan2odom.inverse() * scan2map
    tf::Matrix3x3 tf_R(cos(inv_angle), -sin(inv_angle), 0,
                       sin(inv_angle), cos(inv_angle), 0,          
                       0, 0, 1);
    tf::Vector3 tf_t(inv_t.x(), inv_t.y(), 0);         
    return tf::StampedTransform(tf::Transform(tf_R, tf_t), scan2odom.stamp_, "map", scan2odom.frame_id_);
}

void makePerturbedOdom(
    const Eigen::Vector4d& noise_level, const Eigen::Vector2d& init_pos, Eigen::Vector3d delta_p, Eigen::Vector3d noise, 
    nav_msgs::Odometry& odom, double init_angle, double duration, std::string frame_id, std::string child_id
) {
    /// @note since the input delta_p is in the map frame, no transformation is needed.
    static int cnt = 0;
    static Eigen::Vector3d __pose__ = Eigen::Vector3d(init_pos.x(), init_pos.y(), init_angle);
    static std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());
    static std::normal_distribution<double> trans_noise(0.0, noise_level(0));
    static double last_angle_vel = 0.0;
    Eigen::Matrix2d R, dR;
    dR << cos(delta_p(2)), -sin(delta_p(2)), sin(delta_p(2)), cos(delta_p(2));
    R << cos(__pose__.z()), -sin(__pose__.z()), sin(__pose__.z()), cos(__pose__.z());
    __pose__.block<2, 1>(0, 0) = (R * delta_p.block<2, 1>(0, 0) + __pose__.block<2, 1>(0, 0)).eval();
    R = (R * dR).eval();
    __pose__(2) = atan2(R(1, 0), R(0, 0));
    odom.header.frame_id = frame_id;
    odom.header.seq = cnt;
    odom.header.stamp = ros::Time::now();
    odom.child_frame_id = child_id;
    odom.pose.pose.position.x = __pose__.x() + noise.x();
    odom.pose.pose.position.y = __pose__.y() + noise.y();
    odom.pose.pose.position.z = 0.0;
    const Eigen::Quaterniond qt(Eigen::AngleAxisd(__pose__.z() + noise.z(), Eigen::Vector3d::UnitZ()));
    odom.pose.pose.orientation.w = qt.w();
    odom.pose.pose.orientation.x = qt.x();
    odom.pose.pose.orientation.y = qt.y();
    odom.pose.pose.orientation.z = qt.z();
    for (int i = 0; i < 2; i++)
        odom.pose.covariance[7 * i] = std::pow(noise_level(0), 2);
    odom.pose.covariance.back() = std::pow(noise_level(1), 2);
    if (cnt > 5) {
        double angle_vel = delta_p(2) / duration;
        odom.twist.twist.linear.x = (delta_p(0) / duration) + trans_noise(engine) * 32.;
        odom.twist.twist.linear.y = (delta_p(1) / duration) + trans_noise(engine) * 32.;
        odom.twist.twist.linear.z = 0.0;
        odom.twist.twist.angular.x = 0.0;
        odom.twist.twist.angular.y = 0.0;
        odom.twist.twist.angular.z = 0.95 * angle_vel + 0.05 * last_angle_vel;
        last_angle_vel = odom.twist.twist.angular.z;
        for (int i = 0; i < 2; i++)
            odom.twist.covariance[7 * i] = std::pow(noise_level(2), 2);
        odom.twist.covariance.back() = std::pow(noise_level(3), 2);
    }
    cnt++;
}

void odomTFSimulation(
    const Eigen::Vector4d& noise_level, Eigen::Vector3d delta_p, tf::StampedTransform& tf, 
    Eigen::Vector3d& noise, std::string frame_id, std::string child_id
) {
    static Eigen::Vector3d pose = Eigen::Vector3d::Zero();
    static std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());
    static std::normal_distribution<double> trans_noise(0.0, noise_level(0));
    static std::normal_distribution<double> rot_noise(0.0, noise_level(1));
    noise(0) = trans_noise(engine);
    noise(1) = trans_noise(engine);
    noise(2) = rot_noise(engine);
    delta_p += noise;
    Eigen::Matrix2d R, dR;
    dR << cos(delta_p(2)), -sin(delta_p(2)), sin(delta_p(2)), cos(delta_p(2));
    R << cos(pose.z()), -sin(pose.z()), sin(pose.z()), cos(pose.z());

    pose.block<2, 1>(0, 0) = (R * delta_p.block<2, 1>(0, 0) + pose.block<2, 1>(0, 0)).eval();
    R = (R * dR).eval();
    pose(2) = atan2(R(1, 0), R(0, 0));

    tf::Matrix3x3 tf_R(R(0, 0), R(0, 1), 0,
                       R(1, 0), R(1, 1), 0,
                        0, 0, 1);
    tf::Vector3 tf_t(pose(0), pose(1), 0);
    tf::Transform transform(tf_R, tf_t);
    tf = tf::StampedTransform(transform, ros::Time::now(), frame_id, child_id);
}

void sendTransform(Eigen::Vector3d p, std::string frame_id, std::string child_frame_id) {
    static tf::TransformBroadcaster tfbr;
    tf::StampedTransform transform;
    makeTransform(p, frame_id, child_frame_id, transform);
    tfbr.sendTransform(transform);
}

void sendStampedTranform(const tf::StampedTransform& _tf) {
    static tf::TransformBroadcaster tfbr;
    tfbr.sendTransform(_tf);
}

double pidAngle(const Eigen::Vector2d& orient, const Eigen::Vector2d& obs, double now) {
    Eigen::Vector2d vec = orient - obs;
    double target = atan2(vec.y(), vec.x());
    static double old_diff = 0.0, accum = 0.0;
    double diff = target - now;
    if (now > 2.5 && target < -2.5) {
        diff += 2 * M_PI;
    } else if (now < -2.5 && target > 2.5) {
        diff -= 2 * M_PI;
    }
    double result = K_P * diff + K_I * accum + K_D * (diff - old_diff);
    accum += diff;
    old_diff = diff;
    return result;
}

void makeScan(
    const std::vector<double>& range, const Eigen::Vector3d& angles,
    sensor_msgs::LaserScan& scan, std::string frame_id, double scan_time
) {
    static int cnt = 0;
    scan.ranges.clear();
    for (double val: range)
        scan.ranges.push_back(0.02 * val);
    scan.header.stamp = ros::Time::now();
    scan.header.seq = cnt;
    scan.header.frame_id = frame_id;
    scan.angle_min = angles.x();
    scan.angle_max = angles.y();
    scan.angle_increment = angles.z() * 5.0;
    scan.range_min = 0.02;
    scan.range_max = 100.0;
    scan.scan_time = scan_time;
    scan.time_increment = 1e-9;
    cnt++;
}

// trans 以及 speed 都是小车坐标系的(因为是IMU嘛)
#include <algorithm>
#include <deque>
void makeImuMsg(
    const Eigen::Vector2d& speed,
    std::string frame_id,
    double now_ang,
    double duration,
    sensor_msgs::Imu& msg,
    Eigen::Vector2d vel_var,
    Eigen::Vector2d ang_var
) {
    static int cnt = 0;
    static Eigen::Vector2d last_vel = Eigen::Vector2d::Zero(), last_acc = Eigen::Vector2d::Zero();
    static double last_ang = 0.0, last_ang_vel = 0.0;
    static std::deque<double> durations;
    msg.header.frame_id = cnt;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = frame_id;

    msg.angular_velocity_covariance[0] = ang_var(0);
    msg.angular_velocity_covariance[4] = ang_var(1);

    msg.linear_acceleration_covariance[0] = vel_var(0);
    msg.linear_acceleration_covariance[4] = vel_var(1);
    Eigen::Vector2d acc = Eigen::Vector2d::Zero();
    double ang_vel = 0.0;
    if (cnt > 5) {
        Eigen::Vector2d this_vel = speed / duration;
        this_vel = 0.8 * speed + 0.2 * last_vel;
        acc = (this_vel - last_vel) / duration;
        acc = 0.8 * acc + 0.2 * last_acc;
        last_acc = acc;
        last_vel = this_vel;
        ang_vel = (now_ang - last_ang) / duration;
        ang_vel = 0.95 * ang_vel + 0.05 * last_ang_vel;
        last_ang_vel = ang_vel;
    }
    msg.linear_acceleration.x = acc.x();
    msg.linear_acceleration.y = acc.y();
    msg.linear_acceleration.z = 0.0;
    msg.angular_velocity.x = 0.0;
    msg.angular_velocity.y = 0.0;
    msg.angular_velocity.z = ang_vel;
    cnt++;
}

double makeImuMsg(
    const Eigen::Vector2d& speed,
    std::string frame_id,
    double now_ang,
    double duration,
    sensor_msgs::Imu& msg,
    std::ofstream* file
) {
    makeImuMsg(speed, frame_id, now_ang, duration, msg, Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero());
    if (file != nullptr)
        (*file) << msg.linear_acceleration.x << "," << msg.linear_acceleration.y << "," << duration << std::endl;
    return duration;
}
