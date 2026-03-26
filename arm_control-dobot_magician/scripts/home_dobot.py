import cv2
import numpy as np
import yaml
import os
import sys
from scipy.spatial.transform import Rotation as R

#give time between

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
#from utils import draw_tag_info, get_gripper_to_tag, get_robot_base_to_camera
from control.mydobot import MyDobot , get_dobot_port
from realsense.realsense_init import initialize_pipeline, get_camera_intrinsics, initialize_detector, process_frames, detect_tags
from calibration.utils import draw_tag_info, get_camera_to_tag_matrix, get_robot_base_to_ee, get_average_transformation

# Example usage of the pydobot library
port = get_dobot_port()
device = MyDobot(port=port)
device.set_home(250, 0, 50)
device.home()