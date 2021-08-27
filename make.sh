#！/bin/sh

if [ a"$2" = a ]; then
    catkin_make -j$1 -DCMAKE_BUILD_TYPE=RELEASE
else
    catkin_make -j$1 -DCMAKE_BUILD_TYPE=DEBUG --build build_debug
fi

if [ ! -d ./bags ]; then
    echo "[100%] Creating folder 'bags' for rosbag output."
    mkdir bags
fi