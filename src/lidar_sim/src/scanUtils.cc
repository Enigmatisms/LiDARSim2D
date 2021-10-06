#include <random>
#include "scanUtils.hpp"

double K_P = 0.2;
double K_I = 0.0001;
double K_D = 0.001;
Eigen::Vector3d __pose__;

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
    Eigen::Vector3d delta_p, nav_msgs::Odometry& odom, 
    Eigen::Vector2d noise_level, std::string frame_id, std::string child_id
) {
    /// @note since the input delta_p is in the map frame, no transformation is needed.
    static int cnt = 0;
    static std::default_random_engine engine(std::chrono::system_clock::now().time_since_epoch().count());
    static std::normal_distribution<double> trans_noise(0.0, noise_level(0));
    static std::normal_distribution<double> rot_noise(0.0, noise_level(1));
    delta_p(0) = delta_p(0) * 0.02 + trans_noise(engine);
    delta_p(1) = delta_p(1) * 0.02 + trans_noise(engine);
    delta_p(2) += rot_noise(engine);
    __pose__ = __pose__ + delta_p;
    odom.header.frame_id = frame_id;
    odom.header.seq = cnt;
    odom.header.stamp = ros::Time::now();
    odom.child_frame_id = child_id;
    odom.pose.pose.position.x = __pose__.x();
    odom.pose.pose.position.y = __pose__.y();
    odom.pose.pose.position.z = 0.0;
    Eigen::Quaterniond qt(Eigen::AngleAxisd(__pose__.z(), Eigen::Vector3d::UnitZ()));
    odom.pose.pose.orientation.w = qt.w();
    odom.pose.pose.orientation.x = qt.x();
    odom.pose.pose.orientation.y = qt.y();
    odom.pose.pose.orientation.z = qt.z();
    cnt++;
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
void makeImuMsg(
    const Eigen::Vector2d& speed,
    std::string frame_id,
    double now_ang,
    sensor_msgs::Imu& msg,
    Eigen::Vector2d vel_var,
    Eigen::Vector2d ang_var
) {
    static int cnt = 0;
    static Eigen::Vector2d last_vel = Eigen::Vector2d::Zero(), last_acc = Eigen::Vector2d::Zero();
    static double last_ang = 0.0, last_ang_vel = 0.0;
    static ros::Time last_stamp = ros::Time::now();
    msg.header.frame_id = cnt;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = frame_id;

    msg.angular_velocity_covariance[0] = ang_var(0);
    msg.angular_velocity_covariance[4] = ang_var(1);

    msg.linear_acceleration_covariance[0] = vel_var(0);
    msg.linear_acceleration_covariance[4] = vel_var(1);
    const double duration = (msg.header.stamp - last_stamp).toSec();
    Eigen::Vector2d acc = Eigen::Vector2d::Zero();
    double ang_vel = 0.0;
    if (cnt > 0) {
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
