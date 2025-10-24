# ===== Generate Dataset with RealSense Camera ===== # 


# ===== Third-party Imports ===== #
import cv2
import numpy as np
from tqdm import trange
import hydra
from omegaconf import DictConfig
import torch
import rclpy

# ===== Local application/library scpecific imports ===== #
from stretch_ros2_app import StretchApp
# I got these from concept graph repo
from general_utils import (
    get_exp_out_path, 
    get_stream_data_out_path, 
    measure_time, 
    save_hydra_config, 
    should_exit_early
)

# if you do this datacollection and concept graph on the same computer, hydra config files should match witch concept graph repos
# specifically, streamlined_detections.yaml, realsense.yaml, base_paths.yaml
@hydra.main(
    version_base=None, 
    config_path="hydra_configs/", 
    config_name="streamlined_detections")

def main(cfg : DictConfig):
    
    # --- Initialize ROS 2 ---
    rclpy.init()
    app = None  # Initialize app to None for the finally block
    
    try:
        app = StretchApp() 
        print("StretchApp node initialized. Waiting for data...")

        # Output folder of the detections experiment to use
        det_exp_path = get_exp_out_path(
            dataset_root=cfg.dataset_root, 
            scene_id=cfg.scene_id, 
            exp_suffix=cfg.exp_suffix)

        stream_rgb_path, stream_depth_path, stream_poses_path = get_stream_data_out_path(
            dataset_root=cfg.dataset_root, 
            scene_id=cfg.scene_id,
            make_dir=True)
        
        save_hydra_config(cfg, det_exp_path)

        ## Looping for frames 
        exit_early_flag = False
        frame_idx = 0

        while True:
            # --- Spin the ROS 2 node once to process callbacks ---
            # This will populate self.color_image, self.depth_image, etc.
            rclpy.spin_once(app, timeout_sec=0.1) 
            
            torch.cuda.empty_cache()

            # Check if we should exit early only if the flag hasn't been set yet
            if not exit_early_flag and should_exit_early(cfg.exit_early_file):
                print("Exit early signal detected. Skipping to the final frame...")
                exit_early_flag = True
                break

            # Get the frame data
            s_rgb, s_depth, _, s_camera_pose = app.get_frame_data()

            # Wait until the node has received its first frames
            if s_rgb is None or s_depth is None or s_camera_pose is None:
                if frame_idx == 0: # Only print this on the first few tries
                    print("Waiting for first frame data from ROS topics...")
                cv2.waitKey(100) # Wait a moment before trying again
                continue 
            
            # image_rgb = cv2.cvtColor(s_rgb, cv2.COLOR_BGR2RGB)
            # Show images 
            cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
            cv2.imshow("RealSense", np.asanyarray(s_rgb)) 
            cv2.waitKey(1)

            # Save the rgb to the stream folder with an appropriate name
            color_path = stream_rgb_path / f"{frame_idx}.jpg"
            cv2.imwrite(str(color_path), s_rgb)

            # Save depth to the stream folder with an appropriate name
            curr_stream_depth_path = stream_depth_path / f"{frame_idx}.png"
            cv2.imwrite(str(curr_stream_depth_path), s_depth)
            
            # Save the camera pose to the stream folder with an appropriate name 
            curr_stream_pose_path = stream_poses_path / f"{frame_idx}"
            np.save(str(curr_stream_pose_path), s_camera_pose)

            frame_idx += 1

            ## The second condition for exit 
            if frame_idx == 300:
                print("Reached target frame count (300).")
                break 
    
    finally:
        # --- Clean up ROS 2 ---
        if app is not None:
            app.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()
        print("ROS 2 node shut down and windows closed.")
        

if __name__ == "__main__":
    measure_time(main)()