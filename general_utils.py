import json
import logging
from pathlib import Path
from omegaconf import OmegaConf
import time

def cfg_to_dict(input_cfg):
    """ Convert a Hydra configuration object to a native Python dictionary,
    ensuring all special types (e.g., ListConfig, DictConfig, PosixPath) are
    converted to serializable types for JSON. Checks for non-serializable objects. """
    
    def convert_to_serializable(obj):
        """ Recursively convert non-serializable objects to serializable types. """
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, Path):
            return str(obj)
        return obj

    def check_serializability(obj, context=""):
        """ Attempt to serialize the object, raising an error if not possible. """
        try:
            json.dumps(obj)
        except TypeError as e:
            raise TypeError(f"Non-serializable object encountered in {context}: {e}")

        if isinstance(obj, dict):
            for k, v in obj.items():
                check_serializability(v, context=f"{context}.{k}" if context else str(k))
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                check_serializability(item, context=f"{context}[{idx}]")

    # Convert Hydra configs to native Python types
    # check if its already a dictionary, in which case we don't need to convert it
    if not isinstance(input_cfg, dict):
        native_cfg = OmegaConf.to_container(input_cfg, resolve=True)
    else:
        native_cfg = input_cfg
    # Convert all elements to serializable types
    serializable_cfg = convert_to_serializable(native_cfg)
    # Check for serializability of the entire config
    check_serializability(serializable_cfg)

    return serializable_cfg

def get_stream_data_out_path(dataset_root, scene_id, make_dir=True):
    stream_data_out_path = Path(dataset_root) / scene_id
    stream_rgb_path = stream_data_out_path / "rgb"
    stream_depth_path = stream_data_out_path / "depth"
    stream_poses_path = stream_data_out_path / "poses"
    
    if make_dir:
        stream_rgb_path.mkdir(parents=True, exist_ok=True)
        stream_depth_path.mkdir(parents=True, exist_ok=True)
        stream_poses_path.mkdir(parents=True, exist_ok=True)
        
    return stream_rgb_path, stream_depth_path, stream_poses_path

def get_exp_out_path(dataset_root, scene_id, exp_suffix, make_dir=True):
    exp_out_path = Path(dataset_root) / scene_id / "exps" / f"{exp_suffix}"
    if make_dir:
        exp_out_path.mkdir(exist_ok=True, parents=True)
    return exp_out_path

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        # print(f"Starting {func.__name__}...")
        result = func(*args, **kwargs)  # Call the function with any arguments it was called with
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Done! Execution time of {func.__name__} function: {elapsed_time:.2f} seconds")
        return result  # Return the result of the function call
    return wrapper

def get_exp_config_save_path(exp_out_path, is_detection_config=False):
    params_file_name = "config_params"
    if is_detection_config:
        params_file_name += "_detections"
    return exp_out_path / f"{params_file_name}.json"

def save_hydra_config(hydra_cfg, exp_out_path, is_detection_config=False):
    exp_out_path.mkdir(exist_ok=True, parents=True)
    with open(get_exp_config_save_path(exp_out_path, is_detection_config), "w") as f:
        dict_to_dump = cfg_to_dict(hydra_cfg)
        json.dump(dict_to_dump, f, indent=2)

def should_exit_early(file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        # Check if we should exit early
        if data.get("exit_early", False):
            # Reset the exit_early flag to False
            data["exit_early"] = False
            # Write the updated data back to the file
            with open(file_path, 'w') as file:
                json.dump(data, file)
            return True
        else:
            return False
    except Exception as e:
        # If there's an error reading the file or the key doesn't exist, 
        # log the error and return False
        print(f"Error reading {file_path}: {e}")
        logging.info(f"Error reading {file_path}: {e}")
        return False
