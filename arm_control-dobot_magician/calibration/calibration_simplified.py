import cv2
import numpy as np
import yaml
import os
import sys
from scipy.spatial.transform import Rotation as R

#give time between

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import draw_tag_info, get_gripper_to_tag, get_robot_base_to_camera
from control.mydobot import MyDobot , get_dobot_port
from realsense.realsense_init import initialize_pipeline, get_camera_intrinsics, initialize_detector, process_frames, detect_tags
from calibration.utils import draw_tag_info, get_camera_to_tag_matrix, get_robot_base_to_ee, get_average_transformation
'''
Note: rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth) 
this is the function that converts pixels to depth

'''
def save_transformation(transformation_data, filename):
    with open(filename, "w") as file:
        yaml.dump(transformation_data, file)


def main():
    pipeline, profile, align = initialize_pipeline()
    fx, fy, cx, cy = get_camera_intrinsics(profile)
    detector, tag_size = initialize_detector(0.0792)  # Set the tag size in meters 0.0792
    port = get_dobot_port()
    device = MyDobot(port=port)
    device.home()
    print("Homing the robotic arm for calibration...")
    print("Attach the AprilTag 15.3cm above the gripper and make the x-axes aligned before calibration.")

    cHt_list = []
    bHg_list = []

    print("At the camera window, press Enter to record data, press 'q' to quit and calculate the transformation.")

    while True:
        color_image, depth_image, _, _ = process_frames(pipeline, align)
        if color_image is None or depth_image is None:
            continue

        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        tags = detect_tags(detector, gray_image, [fx, fy, cx, cy], tag_size)

        # cHt, transforms a point in the target frame to the camera frame, which is the pose of the tag in camera frame.
        # bHg, transforms a point in the gripper frame to the base frame, which is the pose of the gripper in base frame.
        if tags:
            tag = tags[0]

            if tag.pose_t is not None:
                cHt = get_camera_to_tag_matrix(tag)
                bHg = get_robot_base_to_ee(device.get_pose().position)
                draw_tag_info(color_image, tag)

        cv2.imshow("AprilTag Detection", color_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("\r") or key == ord("\n"):  # Enter key
            if tags:
                # saving all the matrix
                cHt_list.append(cHt)
                bHg_list.append(bHg)
                print(f"camera to tag: \n{cHt}")
                print(f"gripper to base: \n{bHg}")
                print(f"Data recorded. Total data points: {len(cHt_list)}")

    if len(cHt_list) >= 1:
        # Inverse of cHt --> tHc
        inv_cHt_list = [np.linalg.inv(cHt) for cHt in cHt_list]

        gHt = get_gripper_to_tag()
        print(gHt)
        
        bHc_list = [get_robot_base_to_camera(bHg, gHt, inv_cHt) for bHg, inv_cHt in zip(bHg_list, inv_cHt_list)]
        avg_cHb, avg_cHb_R, avg_cHb_t, avg_rpy = get_average_transformation(bHc_list)
        print(f"Rotation Matrix:\n{avg_cHb_R}")
        print(f"Translation Vector:\n{avg_cHb_t}")
        #print(avg_rpy)

        transformation_data = {"rotation": avg_cHb_R.tolist(), "translation": avg_cHb_t.tolist()}
        save_transformation(transformation_data, os.path.join(os.path.dirname(__file__), "..", "config", "camera_to_base_transformation.yaml"))
    else:
        print("Not enough data points to calculate the transformation.")

    pipeline.stop()
    device.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()