

output_base_path="/home/stn/Dataset/slam_loc/"
for ((i=0;i<6;i++)) do
    for folder in $(find ${output_base_path} -maxdepth 1 -type d -name "hfps${i}*"); do 
        echo "Copying bags/hfps${i}_slam.lgp to ${folder}/slam_initial.lgp"
        cp bags/hfps${i}_slam.lgp ${folder}/slam_initial.lgp
    done
done