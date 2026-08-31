# NB Jetson Deployment

`nb_mouse_bottle_deployment.zip` contains the trained `best.pt`, source code,
training result evidence, and this guide. It does not include the raw photo
set or training virtual environment because they are not required for live
Jetson inference.

## Run on Jetson

Copy the zip to the Jetson home directory and run:

```bash
cd ~
unzip -o nb_mouse_bottle_deployment.zip
source ~/qyy_env/bin/activate
python ~/nb/src/ros2_yolo_node.py --model ~/nb/model/best.pt --camera 0 --conf 0.8 --imgsz 416 --width 640 --height 480 --fps 30 --device 0
```

The node is named `nb_mouse_bottle_yolo_node`; it displays detection boxes and
publishes JSON detections to `/yolo_detections`.

Stop the program with `Ctrl+C`.
