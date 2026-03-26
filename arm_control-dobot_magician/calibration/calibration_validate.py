import cv2
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from control.mydobot import MyDobot, get_dobot_port
from realsense.realsense_init import initialize_pipeline, get_camera_intrinsics, initialize_detector, process_frames, detect_tags
from calibration.utils import load_transformation, draw_tag_info



def main():
    print("Starting calibration validation...")
    pipeline, profile, align = initialize_pipeline()
    fx, fy, cx, cy = get_camera_intrinsics(profile)
    print(f"Camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
    detector, tag_size = initialize_detector(0.0792)  # Set the tag size in meters
    port = get_dobot_port()
    print(f"Connecting to Dobot at port: {port}")
    device = MyDobot(port=port)
    print("Homing the robotic arm for calibration...")
    device.home()
    device.speed(velocity=120, acceleration=120)

    try:
        R_camera2base, t_camera2base = load_transformation(os.path.join(os.path.dirname(__file__), "..", "config", "camera_to_base_transformation.yaml"))
        print("Transformation loaded.")
        print(f"Rotation \n{R_camera2base}")
        print(f"Translation \n{t_camera2base}\n\n")
    except Exception as e:
        print(f"Error loading transformation: {e}")
        pipeline.stop()
        device.close()
        return

    print("Press Enter to record the AprilTag pose. The arm will move to there. Press 'q' to quit.")

    tag_coordinate = None  # Define tag_coordinate outside the loop
    
    while True:
        color_image, depth_image, _, _ = process_frames(pipeline, align)
        if color_image is None or depth_image is None:
            print("Failed to get frames, retrying...")
            continue
        
        # read alarm of robotic arm
        alarm = device.get_alarms()

        if alarm != set():
            # clear alarm
            print("Alarm detected, clear the alarm:", alarm)   
            device.clear_alarms()
        
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        tags = detect_tags(detector, gray_image, [fx, fy, cx, cy], tag_size)

        if tags:
            tag = tags[0]
            if tag.pose_t is not None:
                tag_coordinate = tag.pose_t.copy()  # Make a copy to ensure it's not modified by reference
                tag_coordinate *= 1000
                draw_tag_info(color_image, tag)
                # Update display to show tag was found
                cv2.putText(color_image, "Tag Found!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        else:
            # Show message if no tag is found
            cv2.putText(color_image, "No Tag Found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            tag_coordinate = None

        cv2.imshow("AprilTag Detection", color_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("Quitting...")
            break
        elif key == ord("h"):
            # Go home
            print("Homing device...")
            device.home()
        elif key == ord("\r") or key == ord("\n"):  # Enter key
            print("Enter key pressed")
            if tags and tag_coordinate is not None:
                print(f"Tag coordinate in camera frame: \n{tag_coordinate}")
                try:
                    target_coordinate = R_camera2base @ tag_coordinate + t_camera2base
                    print(f"Moving to target point: \n{target_coordinate[0], target_coordinate[1], target_coordinate[2]}")
                    # offset the z-coordinate to avoid collision
                    device.move_to(target_coordinate[0][0], target_coordinate[1][0], target_coordinate[2][0] )
                except Exception as e:
                    print(f"Error moving to target: {e}")
            else:
                print("No tag detected, cannot move.")

    print("Cleaning up resources...")
    pipeline.stop()
    device.close()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == "__main__":
    main()