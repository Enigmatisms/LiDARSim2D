# 五种参数激光雷达配置，6个bag
# (240度，0.1, 0.25, 0.5) (240度， 0.25度，噪声0.08) （180度，0.25度分辨率）

source /opt/ros/noetic/setup.bash 
source /home/stn/slam/LiDARSim2D/devel/setup.bash 

cat config/automatic.conf | while IFS=' ' read range noise resolution auto; do 
    echo "Range: ${range}, noise std: ${noise}, resolution: ${resolution}, auto: ${auto}";
    for ((i=0;i<6;i++)); do
        roslaunch lidar_sim scan_auto.launch bag_name:="hfps$i" map_name:="standard$i" angle_incre:=${resolution} angle_span:=${range} lidar_noise:=${noise} use_recorded_path:=${auto}
    done
done