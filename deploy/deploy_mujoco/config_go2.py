from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
import yaml


class Config:
    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.msg_type = config["msg_type"]
            self.imu_type = config["imu_type"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            self.policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)

            self.joint2motor_idx = config["joint2motor_idx"]
            self.kps = np.array(config["kps"], dtype=np.float32)
            self.kds = np.array(config["kds"], dtype=np.float32)
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)

            self.obs_scales_ang_vel = config["obs_scales_ang_vel"]
            self.obs_scales_dof_pos = config["obs_scales_dof_pos"]
            self.obs_scales_dof_vel = config["obs_scales_dof_vel"]
            self.action_scale = config["action_scale"]
            self.command_scale = np.array(config["command_scale"], dtype=np.float32)

            self.lin_vel_deadband = config["lin_vel_deadband"]
            self.ang_vel_deadband = config["ang_vel_deadband"]
            self.cmd_x_range = config["cmd_x_range"]
            self.cmd_y_range = config["cmd_y_range"]
            self.cmd_yaw_range = config["cmd_yaw_range"]

            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            self.history_length = config["history_length"]
