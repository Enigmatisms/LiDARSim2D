#include <random>
#include "utils/scanUtils.hpp"

double K_P = 0.2;
double K_I = 0.0001;
double K_D = 0.001;

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

void makePerturbedOdom(
    const Eigen::Vector4d& noise_level, Eigen::Vector3d delta_p, nav_msgs::Odometry& odom, 
    double duration, std::string frame_id, std::string child_id
) {
    /// @note since the input delta_p is in the map frame, no transformation is needed.
    static int cnt = 0;
    static Eigen::Vector3d __pose__ = Eigen::Vector3d::Zero();
    static std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());
    static std::normal_distribution<double> trans_noise(0.0, noise_level(0));
    static std::normal_distribution<double> rot_noise(0.0, noise_level(1));
    static double last_angle_vel = 0.0;
    delta_p(0) = delta_p(0) + trans_noise(engine);
    delta_p(1) = delta_p(1) + trans_noise(engine);
    delta_p(2) += rot_noise(engine);
    Eigen::Matrix2d R, dR;
    dR << cos(delta_p(2)), -sin(delta_p(2)), sin(delta_p(2)), cos(delta_p(2));
    R << cos(__pose__.z()), -sin(__pose__.z()), sin(__pose__.z()), cos(__pose__.z());
    R = (R * dR).eval();
    __pose__.block<2, 1>(0, 0) = (R * delta_p.block<2, 1>(0, 0) + __pose__.block<2, 1>(0, 0)).eval();
    __pose__(2) = atan2(R(1, 0), R(0, 0));
    odom.header.frame_id = frame_id;
    odom.header.seq = cnt;
    odom.header.stamp = ros::Time::now();
    odom.child_frame_id = child_id;
    odom.pose.pose.position.x = __pose__.x();
    odom.pose.pose.position.y = __pose__.y();
    odom.pose.pose.position.z = 0.0;
    const Eigen::Quaterniond qt(Eigen::AngleAxisd(__pose__.z(), Eigen::Vector3d::UnitZ()));
    odom.pose.pose.orientation.w = qt.w();
    odom.pose.pose.orientation.x = qt.x();
    odom.pose.pose.orientation.y = qt.y();
    odom.pose.pose.orientation.z = qt.z();
    for (int i = 0; i < 2; i++)
        odom.pose.covariance[7 * i] = std::pow(noise_level(0), 2);
    odom.pose.covariance.back() = std::pow(noise_level(1), 2);
    if (cnt > 5) {
        double angle_vel = delta_p(2) / duration;
        odom.twist.twist.linear.x = (delta_p(0) / duration) + trans_noise(engine);
        odom.twist.twist.linear.y = (delta_p(1) / duration) + trans_noise(engine);
        odom.twist.twist.linear.y = 0.0;
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
    std::string frame_id, std::string child_id
) {
    static Eigen::Vector3d pose = Eigen::Vector3d::Zero();
    static std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());
    static std::normal_distribution<double> trans_noise(0.0, noise_level(0));
    static std::normal_distribution<double> rot_noise(0.0, noise_level(1));
    delta_p(0) = delta_p(0) + trans_noise(engine);
    delta_p(1) = delta_p(1) + trans_noise(engine);
    delta_p(2) += rot_noise(engine);
    Eigen::Matrix2d R, dR;
    dR << cos(delta_p(2)), -sin(delta_p(2)), sin(delta_p(2)), cos(delta_p(2));
    R << cos(pose.z()), -sin(pose.z()), sin(pose.z()), cos(pose.z());
    R = (R * dR).eval();
    pose.block<2, 1>(0, 0) = (R * delta_p.block<2, 1>(0, 0) + pose.block<2, 1>(0, 0)).eval();
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
std::pair<double, double> makeImuMsg(
    const Eigen::Vector2d& speed,
    std::string frame_id,
    double now_ang,
    sensor_msgs::Imu& msg,
    Eigen::Vector2d vel_var,
    Eigen::Vector2d ang_var
) {
    static int cnt = 0;
    static Eigen::Vector2d last_vel = Eigen::Vector2d::Zero(), last_acc = Eigen::Vector2d::Zero();
    static double last_ang = 0.0, last_ang_vel = 0.0, last_duration = 0.0;
    static ros::Time last_stamp = ros::Time::now();
    static std::deque<double> durations;
    msg.header.frame_id = cnt;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = frame_id;

    msg.angular_velocity_covariance[0] = ang_var(0);
    msg.angular_velocity_covariance[4] = ang_var(1);

    msg.linear_acceleration_covariance[0] = vel_var(0);
    msg.linear_acceleration_covariance[4] = vel_var(1);
    double duration_raw = (msg.header.stamp - last_stamp).toSec();
    double duration = 0.4 * duration_raw + 0.6 * last_duration;
    durations.push_back(duration);
    if (durations.size() > 5) {
        durations.pop_front();
        std::vector<double> vec(3);
        std::partial_sort_copy(durations.begin(), durations.end(), vec.begin(), vec.end());
        duration = vec.back();
    }
    last_duration = duration;
    last_stamp = msg.header.stamp;
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
    
    return std::make_pair(duration, duration_raw);
}

double makeImuMsg(
    const Eigen::Vector2d& speed,
    std::string frame_id,
    double now_ang,
    sensor_msgs::Imu& msg,
    std::ofstream* file
) {
    const auto& [duration, raw_du] = makeImuMsg(speed, frame_id, now_ang, msg, Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero());
    if (file != nullptr)
        (*file) << msg.linear_acceleration.x << "," << msg.linear_acceleration.y << "," << duration << "," << raw_du << std::endl;
    return duration;
}
