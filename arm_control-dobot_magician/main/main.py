#!/usr/bin/env python3
import cv2
import numpy as np
import pyrealsense2 as rs
import os
import sys
import math
from ultralytics import YOLO
import time

# Add parent directory for Dobot and calibration
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from control.mydobot import MyDobot, get_dobot_port
from calibration.utils import load_transformation

NUT=0
BOLT=1
SCREW=2

# ==================== Helpers (NEW) ====================

def clamp(v, lo, hi): return max(lo, min(hi, v))

def normalize_deg(a):
    """[-180, 180)"""
    return (a + 180.0) % 360.0 - 180.0

def draw_angle_arrow(img, cx, cy, theta_deg, length=50, color=(0,255,255)):
    th = math.radians(theta_deg)
    x2 = int(round(cx + length*math.cos(th)))
    y2 = int(round(cy + length*math.sin(th)))
    cv2.arrowedLine(img, (int(cx),int(cy)), (x2,y2), color, 2, tipLength=0.25)

def pixel_to_3d_with_fallback(x, y, depth_frame, depth_intrinsics, fallback_depth=None):
    if x < 0 or x >= depth_intrinsics.width or y < 0 or y >= depth_intrinsics.height:
        return None
    depth = depth_frame.get_distance(x, y)
    if depth <= 0.2 or depth > 2.0:
        if fallback_depth is None or fallback_depth <= 0.2 or fallback_depth > 2.0:
            return None
        depth = fallback_depth
    pt = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
    return np.array(pt)

def min_area_rect_angle_from_bbox(bgr_img, xyxy):
    """
    Returns (cx, cy, theta_img_deg, aspect_ratio, ok)
    theta_img_deg: major-axis direction in image coords (right=0°, down=+90°).
    """
    x1, y1, x2, y2 = [int(v) for v in xyxy]
    H, W = bgr_img.shape[:2]
    x1, y1 = clamp(x1,0,W-1), clamp(y1,0,H-1)
    x2, y2 = clamp(x2,0,W-1), clamp(y2,0,H-1)

    pad = int(0.15 * max(x2-x1, y2-y1))
    rx1 = clamp(x1 - pad, 0, W-1); ry1 = clamp(y1 - pad, 0, H-1)
    rx2 = clamp(x2 + pad, 0, W-1); ry2 = clamp(y2 + pad, 0, H-1)
    roi = bgr_img[ry1:ry2, rx1:rx2]
    if roi.size == 0: return None, None, None, None, False

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), iterations=2)

    cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None, None, None, None, False
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 50: return None, None, None, None, False

    rect = cv2.minAreaRect(c)  # ((cx,cy),(W,H),ang in (-90,0])
    (rcx, rcy), (RW, RH), ang = rect
    cx_full, cy_full = rx1 + rcx, ry1 + rcy

    # Convert to major-axis angle, normalize
    if RW < RH:
        theta = ang 
        majW, majH = RH, RW
    else:
        theta = ang +90
        majW, majH = RW, RH

    aspect_ratio = (majW / max(majH,1e-6)) if majH > 0 else 999.0
    theta = normalize_deg(theta)
    return cx_full, cy_full, theta, aspect_ratio, True

def compute_world_yaw_from_img_angle(cx, cy, theta_img_deg, depth_frame, depth_intrinsics, R, t, step_px):
    """
    Take two pixels along θ, deproject both, transform to base, atan2(dy, dx) → yaw_world_deg
    """
    th = math.radians(theta_img_deg)
    x2 = clamp(int(round(cx + step_px*math.cos(th))), 0, depth_intrinsics.width - 1)
    y2 = clamp(int(round(cy + step_px*math.sin(th))), 0, depth_intrinsics.height - 1)

    center_depth = depth_frame.get_distance(int(cx), int(cy))
    p1_cam = pixel_to_3d_with_fallback(int(cx), int(cy), depth_frame, depth_intrinsics, None)
    p2_cam = pixel_to_3d_with_fallback(x2, y2, depth_frame, depth_intrinsics, center_depth)
    if p1_cam is None or p2_cam is None:
        return None

    p1 = (R @ p1_cam.reshape(3,1) + t).flatten()
    p2 = (R @ p2_cam.reshape(3,1) + t).flatten()
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return None
    return normalize_deg(math.degrees(math.atan2(dy, dx)))

def safe_move_to(dobot, x_mm, y_mm, z_mm, yaw_deg=None):
    """
    Try common Dobot signatures with yaw; fall back to XYZ-only.
    """
    if yaw_deg is not None:
        '''try:
            #print(1) 
            return dobot.move_to(x_mm, y_mm, z_mm, yaw_deg+90.0)
        except (TypeError, AttributeError): pass'''
        try: 
            #print(2)
            return dobot.move_to(x_mm, y_mm, z_mm, yaw_deg)
        except (TypeError, AttributeError): pass
        try: 
            #print(3)
            return dobot.move_to(x_mm, y_mm, z_mm, yaw_deg)
        except Exception: pass

    print(4)
    return dobot.move_to(x_mm, y_mm, z_mm)

def pixel_to_3d(x, y, depth_frame, depth_intrinsics):
    if x < 0 or x >= depth_intrinsics.width or y < 0 or y >= depth_intrinsics.height:
        return None
    depth = depth_frame.get_distance(x, y)
    point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
    return np.array(point_3d)

def is_within_workspace(coordinates, x_range=(-0.32, 0.32), y_range=(-0.32, 0.32), z_range=(0, 0.15)):
    return True

def get_sorted_detection(results, model):
    sorted_boxes = list()
    for r in results:
        boxes = r.boxes
        if not boxes:
            continue
        for box in boxes:
            conf = float(box.conf.cpu().numpy())
            if conf < 0.35:
                continue
            cls_id = int(box.cls.cpu().numpy())
            cls_name = model.names[cls_id]
            xyxy = box.xyxy[0].cpu().numpy()
            x_center = (xyxy[0] + xyxy[2]) / 2
            y_center = (xyxy[1] + xyxy[3]) / 2
            best = {
                "conf": conf,
                "cls_name": cls_name,
                "x": x_center,
                "y": y_center,
                "xyxy": xyxy,        
            }
            sorted_boxes.append(best)
    
    sorted_boxes.sort(key=lambda x: x["conf"], reverse=True)
    n=len(sorted_boxes)
    if(n>10):
        sorted_boxes=sorted_boxes[0:int(6/10*n)]
    elif(n>5):
        sorted_boxes=sorted_boxes[0:int(8/10*n)]
    return sorted_boxes

def to_class_num(class_name):
    if class_name == "Bolt":  return BOLT
    if class_name == "Nut":   return NUT
    if class_name == "Screw": return SCREW

# ==================== MAIN ====================

def main():
    port = get_dobot_port()
    dobot = MyDobot(port=port)
    dobot.home()
    # 1. YOLO
    model = YOLO("Nuts_Bolts_Screws_yolo.pt")

    
    # 2. RealSense
    ctx = rs.context()
    if not ctx.query_devices():
        print("No RealSense device!")
        return
    dev = ctx.query_devices()[0]
    serial = dev.get_info(rs.camera_info.serial_number)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    #we'll refresh intrinsics each loop from the aligned depth_frame

    # 3. Dobot
    

    # 4. Transform
    R, t = load_transformation("config/camera_to_base_transformation.yaml")
    t = t / 1000.0  # mm → m
    #this part might be wrong. check once by removing 1000.0

    # 5. UI
    cv2.namedWindow("Robot Arm Competition")
    print("Running. Press 'q' to quit, 'h' to home.")
    drop_pos = []

    input("record drop position of Nut");   drop_pos.append(dobot.get_pose().position)

    input("record drop position of Bolt");  drop_pos.append(dobot.get_pose().position)
    input("record drop position of Screw"); drop_pos.append(dobot.get_pose().position)
    '''drop_pos.append([84.35523223876953, -230.95184326171875, -12.855583190917969])
    drop_pos.append([185.82383728027344, -158.1154327392578, -12.141990661621094])
    drop_pos.append([238.30770874023438, -77.67424774169922, -18.17346954345703])'''


    # Optional global yaw offset if your wrist zero is rotated
    #YAW_OFFSET_DEG = 0.0
    dobot.move_to(drop_pos[SCREW][0], drop_pos[SCREW][1], drop_pos[SCREW][2] + 50)

    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # IMPORTANT: intrinsics of the ALIGNED depth frame
        depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()

        color_img = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_RGB2BGR)

        # YOLO
        results = model(color_img, verbose=False)

        sorted_boxes = get_sorted_detection(results, model)
        annotated = results[0].plot() if results else color_img.copy()

        for best in sorted_boxes:
            print("in for")
            if best:
                x_pix, y_pix = int(best["x"]), int(best["y"])
                cv2.circle(annotated, (x_pix, y_pix), 6, (0, 255, 0), -1)
                cv2.putText(annotated, f"{best['cls_name']} {best['conf']:.2f}",
                            (x_pix, y_pix - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # ---------- ANGLE (θ) + WORLD YAW ----------
                theta_img_deg = None
                yaw_world_deg = None
                aspect_ratio = None

                if best.get("xyxy") is not None:
                    cx, cy, theta_img_deg, aspect_ratio, ok = min_area_rect_angle_from_bbox(color_img, best["xyxy"])
                    if ok and theta_img_deg is not None:
                        draw_angle_arrow(annotated, cx, cy, theta_img_deg, 50, (0,255,255))
                        # dynamic step along axis (reduces jitter)
                        x1,y1,x2,y2 = best["xyxy"]
                        step_px = int(0.3 * max(x2 - x1, y2 - y1))
                        step_px = clamp(step_px, 12, 40)
                        yaw_world_deg = compute_world_yaw_from_img_angle(
                            cx, cy, theta_img_deg, depth_frame, depth_intrinsics, R, t, step_px
                        )

                # ---------- POSITION ----------
                point_3d_cam = pixel_to_3d(x_pix, y_pix, depth_frame, depth_intrinsics)
                if point_3d_cam is not None:
                    p_base = R @ point_3d_cam.reshape(3, 1) + t
                    x_b, y_b, z_b = p_base.flatten()

                    # Decide gripper_yaw (parallel jaws close across short axis)
                    gripper_yaw = None
                    cls = best['cls_name']
                    if yaw_world_deg is not None:
                        # Nuts ≈ near-square: fixed yaw; elongated parts: yaw + 90°
                        if (cls == "Nut") or (aspect_ratio is not None and aspect_ratio < 1.3):
                            gripper_yaw = normalize_deg(0.0)
                        else:
                            gripper_yaw = normalize_deg(yaw_world_deg + 90.0)

                    print(f"Best: {cls} | Conf: {best['conf']:.3f} | "
                        f"Base: ({x_b:.3f}, {y_b:.3f}, {z_b:.3f}) m | "
                        f"θ_img: {theta_img_deg if theta_img_deg is not None else 'NA'}° | "
                        f"yaw_world: {yaw_world_deg if yaw_world_deg is not None else 'NA'}° | "
                        f"gripper_yaw: {gripper_yaw if gripper_yaw is not None else 'NA'}°")

                    '''drop_pos[SCREW][0]-70.0 < x_b*1000<drop_pos[SCREW][0]+100.0 and drop_pos[SCREW][1]-70.0<'''

                    try: 
                        if(not(y_b*1000<0)):
                            cn = to_class_num(cls)
                            dobot.grip(enable=False)

                            # Approach with yaw (if available)
                            safe_move_to(dobot, x_b*1000, y_b*1000, 30, yaw_deg=gripper_yaw)

                            alarm = dobot.get_alarms()
                            if alarm:
                                print("Clearing alarm:", alarm)
                                dobot.clear_alarms()
                                continue #check here if it is actually continuing
                            
                            print("The program continues")

                            # Descend
                            safe_move_to(dobot, x_b*1000, y_b*1000, -30, yaw_deg=gripper_yaw)
                            
                            #turn by 90 degrees if class is nuts so that there is a higher chance it moves to the middle
                            if(cn==NUT):
                                safe_move_to(dobot, x_b*1000, y_b*1000, -30, yaw_deg=(gripper_yaw+180)%360)

                            # Grip
                            dobot.grip(enable=True)
                            time.sleep(1)

                            # Lift
                            safe_move_to(dobot, x_b*1000, y_b*1000, 30, yaw_deg=gripper_yaw)

                            # Drop to bin by class
                            
                            dobot.move_to(drop_pos[cn][0], drop_pos[cn][1], 50)
                            dobot.move_to(drop_pos[cn][0], drop_pos[cn][1], -25)

                            dobot.grip(enable=False)
                            time.sleep(0.5)
                            dobot.move_to(drop_pos[cn][0], drop_pos[cn][1],  50)

                            print(f"MOVED → ({x_b*1000:.1f}, {y_b*1000:.1f}, {z_b*1000+30:.1f}) mm")
                    except Exception as e:
                        print("Move failed:", e)
                else:
                    print("Invalid depth at center")

            cv2.imshow("Robot Arm Competition", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('h'):
                dobot.home()

            # Clear alarms
            alarm = dobot.get_alarms()
            if alarm:
                print("Clearing alarm:", alarm)
                dobot.clear_alarms()
            
            time.sleep(0.1)
        time.sleep(1)

    pipeline.stop()
    dobot.close()
    cv2.destroyAllWindows()

if __name__== "__main__":
    main()