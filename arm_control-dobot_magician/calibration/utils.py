import cv2
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

def load_transformation(filename):
    with open(filename, "r") as file:
        transformation = yaml.safe_load(file)
    rotation = np.array(transformation["rotation"])
    translation = np.array(transformation["translation"])
    if translation.shape != (3, 1):
        translation = translation.reshape(3, 1)
    return rotation, translation

def draw_tag_info(color_image, tag):
    for idx in range(len(tag.corners)):
        cv2.line(color_image, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0), 2)
    cv2.circle(color_image, tuple(tag.center.astype(int)), 5, (0, 0, 255), -1)
    center_x, center_y = tag.center.astype(int)
    cv2.putText(color_image, f"ID: {tag.tag_id}", (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def draw_tag_info_with_data(color_image, tag, depth_camera, depth_tag, depth_error, orientation):
    for idx in range(len(tag.corners)):
        cv2.line(color_image, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0), 2)
    cv2.circle(color_image, tuple(tag.center.astype(int)), 5, (0, 0, 255), -1)
    center_x, center_y = tag.center.astype(int)
    cv2.putText(color_image, f"ID: {tag.tag_id}", (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print(f"Tag ID: {tag.tag_id}, Center: {tag.center}, Depth: {depth_tag}, Depth Error: {depth_error}")
    print(f"Orientation (radians): Roll: {orientation[0]}, Pitch: {orientation[1]}, Yaw: {orientation[2]}")

def get_camera_to_tag_matrix(tag):
    """
    Build a 4x4 transformation matrix (camera to tag) in mm from a detected tag object.
    """
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = tag.pose_R
    transformation_matrix[:3, 3] = tag.pose_t.flatten() * 1000
    return transformation_matrix

def get_robot_base_to_ee(pose):
    theta=np.arctan2(pose.y, pose.x)

    robot_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0, pose.x],
            [np.sin(theta), np.cos(theta), 0, pose.y],
            [0, 0, 1, pose.z],
            [0, 0, 0, 1]
        ])
    return robot_matrix

def get_gripper_to_tag():
    # Replace to your measurement here
    gripper_to_tag_matrix = np.array([[-1, 0, 0, 30],    #it's possible it is not -1, but 1
                                       [0, 1, 0, 0],
                                       [0, 0, -1, 153],
                                       [0, 0, 0, 1]], dtype=np.float32)
    return gripper_to_tag_matrix

def get_robot_base_to_camera(T_base_to_ee, T_ee_to_tag, T_tag_to_camera):
    # Paste your implementation here
    T_base_to_camera = T_base_to_ee @ T_ee_to_tag @ T_tag_to_camera
    return T_base_to_camera

def get_average_transformation(mats):
    """
    Compute the average transformation (rotation and translation) from a list of 4x4 matrices.
    Returns: avg_mat, avg_R, avg_t, avg_rpy
    """
    R_mats = [mat[:3, :3] for mat in mats]
    rpys = [R.from_matrix(R_mat).as_euler("xyz", degrees=True) for R_mat in R_mats]
    avg_rpy = np.mean(rpys, axis=0)
    avg_R = R.from_euler("xyz", avg_rpy, degrees=True).as_matrix()
    t_mats = [mat[:3, 3] for mat in mats]
    avg_t = np.mean(t_mats, axis=0)
    avg_mat = np.eye(4)
    avg_mat[:3, :3] = avg_R
    avg_mat[:3, 3] = avg_t
    return avg_mat, avg_R, avg_t, avg_rpy

def get_inverse_transform(H):
    r=H[0:3, 0:3]
    t=H[0:3, 3]

    R_inv=r.T
    t_inv= -R_inv@t

    H_inv=np.eye(4)
    H_inv[0:3, 0:3]=R_inv
    H_inv[0:3, 3]=t_inv
    
    return H_inv


#possible place of error
def rotation_to_euler(R):
    return np.array([
        np.arcsin(-R[2,0]), 
        np.arctan2(R[0,0], R[1,0]),
        np.arctan2(R[2,1], R[2,2])
        ])

def euler_to_rotation(euler):
    R=np.eye(3)
    R[0,0]=np.cos(euler[0])*np.cos(euler[1])
    R[0,1]=-np.cos(euler[2])*np.sin(euler[0])+np.sin(euler[1])*np.sin(euler[2])*np.cos(euler[0])
    R[0,2]=np.sin(euler[0])*np.sin(euler[2])+np.sin(euler[1])*np.cos(euler[2])*np.cos(euler[0])
    R[1,0]=np.sin(euler[0])*np.cos(euler[0])
    R[1,1]=np.sin(euler[0])*np.sin(euler[1])*np.sin(euler[2])+np.cos(euler[2])*np.cos(euler[0])
    R[1,2]=np.sin(euler[0])*np.sin(euler[2])*np.cos(euler[2])-np.sin(euler[2])*np.cos(euler[0])
    R[2,0]=-np.sin(euler[1])
    R[2,1]=np.sin(euler[2])*np.cos(euler[1])
    R[2,2]=np.cos(euler[1])*np.cos(euler[2])

    return R