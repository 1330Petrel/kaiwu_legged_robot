from legged_gym import LEGGED_GYM_ROOT_DIR
import numpy as np
import time
import yaml
import torch

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_

from unitree_sdk2py.utils.crc import CRC

from common.command_helper import create_zero_cmd, create_damping_cmd
from common.rotation_helper import get_gravity_orientation

# DDS 话题名称常量
TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_WIRELESS = "rt/wirelesscontroller"

# 控制层级与停止标志
HIGHLEVEL = 0xEE
LOWLEVEL = 0xFF
TRIGERLEVEL = 0xF0
PosStopF = 2.146e9
VelStopF = 16000.0

# ----------  通信域与接口  ----------
DOMAIN_ID = 1  # 与 unitree_mujoco.py 保持一致
INTERFACE = "lo"  # 使用本地回环接口

ChannelFactoryInitialize(DOMAIN_ID, INTERFACE)  # 初始化 DDS 工厂以建立通道


def init_cmd_go2(cmd: LowCmdGo):
    """按照 GO2 协议初始化 LowCmd 头部及 12 个电机的默认状态"""
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for i in range(12):
        # 置入安全初值，先断开位置/速度控制，只维持零力矩
        cmd.motor_cmd[i].mode = 0x0A  # 0x01
        cmd.motor_cmd[i].q = PosStopF
        cmd.motor_cmd[i].dq = VelStopF  # or qd
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0


class Controller:
    def __init__(self, config: dict) -> None:
        """加载策略、建立通信，并准备控制缓存"""
        self.use_remote_controller = True
        self._crc = CRC()

        self.policy = torch.jit.load(config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR))
        self.policy.eval()

        num_obs = config["num_obs"]
        num_actions = config["num_actions"]
        self.qj = np.zeros(num_actions, dtype=np.float32)
        self.dqj = np.zeros(num_actions, dtype=np.float32)
        self.action = np.zeros(num_actions, dtype=np.float32)
        self.target_dof_pos = np.zeros(num_actions, dtype=np.float32)
        self.obs = np.zeros(num_obs, dtype=np.float32)
        self.cmd = np.array([0.0, 0, 0], dtype=np.float32)

        self.qj_obs = np.zeros(num_actions, dtype=np.float32)
        self.dqj_obs = np.zeros(num_actions, dtype=np.float32)
        # 去掉指令 3 维，历史包含当前帧
        self.obs_without_command = np.zeros(num_obs - 3, dtype=np.float32)
        self.proprio_history = np.zeros((config["history_length"] + 1, num_obs - 3), dtype=np.float32)
        
        # 预分配 Tensor
        self.obs_tensor = torch.zeros((1, num_obs), dtype=torch.float32)
        self.history_tensor = torch.zeros((1, (config["history_length"] + 1) * (num_obs - 3) + 3), dtype=torch.float32)

        self._parse_config(config)
        self._warm_up()

        # 构建低层指令通道
        self.low_cmd = unitree_go_msg_dds__LowCmd_()
        self.low_state = unitree_go_msg_dds__LowState_()
        self.lowcmd_publisher = ChannelPublisher(config["lowcmd_topic"], LowCmdGo)
        self.lowcmd_publisher.Init()
        # ----------  订阅低层状态与遥控器 ----------
        self.wireSub = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
        self.wireSub.Init(self.WirelessControllerHandler, 10)
        self.latest_low_state = None  # 缓存低层状态
        self.latest_wc = None  # 缓存遥控器信息
        self.lowSub = ChannelSubscriber("rt/lowstate", LowStateGo)
        self.lowSub.Init(self.LowStateHandler, 10)  # 注册低层状态回调

        self.keys = 0  # 无线控制器按键位图
        self.lx = self.ly = self.rx = 0.0

        init_cmd_go2(self.low_cmd)

    def _parse_config(self, config: dict) -> None:
        self.control_dt = config["control_dt"]

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
        cmd_forward_range = config["cmd_forward_range"]
        cmd_backward_range = config["cmd_backward_range"]
        cmd_y_range = config["cmd_y_range"]
        cmd_yaw_range = config["cmd_yaw_range"]
        self._cmd_x_min = cmd_forward_range[0]
        self._cmd_y_min = cmd_y_range[0]
        self._cmd_yaw_min = cmd_yaw_range[0]
        self._cmd_forward_scale = (cmd_forward_range[1] - cmd_forward_range[0]) / (1 - self.lin_vel_deadband)
        self._cmd_backward_scale = (cmd_backward_range[1] - cmd_backward_range[0]) / (1 - self.lin_vel_deadband)
        self._cmd_y_scale = (cmd_y_range[1] - cmd_y_range[0]) / (1 - self.lin_vel_deadband)
        self._cmd_yaw_scale = (cmd_yaw_range[1] - cmd_yaw_range[0]) / (1 - self.ang_vel_deadband)

        dof_pos_protect_ratio = config["dof_pos_protect_ratio"]
        dof_pos_redundancy_ratio = config["dof_pos_redundancy_ratio"]
        joint_limits_high = np.array(config["joint_limits_high"], dtype=np.float32)
        joint_limits_low = np.array(config["joint_limits_low"], dtype=np.float32)
        joint_pos_mid = (joint_limits_high + joint_limits_low) / 2.0
        joint_pos_range = (joint_limits_high - joint_limits_low) / 2.0
        self._pos_warn_high = joint_pos_mid + dof_pos_redundancy_ratio * joint_pos_range
        self._pos_warn_low = joint_pos_mid - dof_pos_redundancy_ratio * joint_pos_range
        self._pos_protect_high = joint_pos_mid + dof_pos_protect_ratio * joint_pos_range
        self._pos_protect_low = joint_pos_mid - dof_pos_protect_ratio * joint_pos_range

        torque_limits = np.array(config["torque_limits"], dtype=np.float32)
        self._torque_kp_ratio = torque_limits / self.kps

    def _warm_up(self):
        """对 TorchScript 策略执行若干次推理，避免第一次运行抖动"""
        obs = torch.ones_like(self.obs_tensor)
        history = torch.ones_like(self.history_tensor)
        for _ in range(10):
            _ = self.policy(obs, history)
        print("Network has been warmed up.")

    def LowStateHandler(self, msg: LowStateGo):
        # 仅缓存消息避免在回调中耗时处理
        self.latest_low_state = msg

    def WirelessControllerHandler(self, msg: WirelessController_):
        # 保存最新遥控器状态供主循环读取
        self.latest_wc = msg
        self.keys = msg.keys
        self.lx = msg.lx
        self.ly = msg.ly
        self.rx = msg.rx

    def send_cmd(self, cmd: LowCmdGo):
        """在发送前补齐 CRC，确保指令被底层接受"""
        cmd.crc = self._crc.Crc(cmd)
        self.lowcmd_publisher.Write(cmd)

    def zero_torque_state(self):
        """零力矩待机：持续发送零力矩指令，等待遥控器 start 键启动"""
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while not (self.keys >> 2) & 1:  # 检测 Start 键作为安全确认
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

    def move_to_default_pos(self):
        """缓慢插值到默认关节角，确保机器人从安全姿态起步"""
        print("Moving to default pos.")
        total_time = 2
        num_step = int(total_time / self.control_dt)
        kp_min, kp_max = 5.0, 40.0
        kd_min, kd_max = 0.2, 0.6

        dof_idx = self.joint2motor_idx

        init_dof_pos = np.zeros(12, dtype=np.float32)
        for i in range(12):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

        # 按照线性插值平滑过渡至默认角度
        for i in range(num_step):
            alpha = i / num_step
            kp = kp_min + (kp_max - kp_min) * alpha
            kd = kd_min + (kd_max - kd_min) * alpha
            for j in range(12):
                motor_idx = dof_idx[j]
                target_pos = self.default_angles[j]
                self.low_cmd.motor_cmd[motor_idx].q = init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = kp
                self.low_cmd.motor_cmd[motor_idx].kd = kd
                self.low_cmd.motor_cmd[motor_idx].tau = 0.0
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

    def default_pos_state(self):
        """保持默认姿态并等待遥控器 A 键，防止外部扰动"""
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        while not (self.keys >> 8) & 1:
            for i in range(12):
                motor_idx = self.joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].dq = 0.0
                self.low_cmd.motor_cmd[motor_idx].kp = 40.0
                self.low_cmd.motor_cmd[motor_idx].kd = 0.6
                self.low_cmd.motor_cmd[motor_idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.control_dt)

    def _process_remote_command(self):
        """处理遥控器输入，应用死区和范围映射"""
        # 从 [-1, -deadband] U [deadband, 1] 映射到 [-max, -min] U [min, max]
        # left-y for forward/backward
        ly = self.ly
        if ly > self.lin_vel_deadband:
            self.cmd[0] = self._cmd_x_min + (ly - self.lin_vel_deadband) * self._cmd_forward_scale
        elif ly < -self.lin_vel_deadband:
            self.cmd[0] = -self._cmd_x_min + (ly + self.lin_vel_deadband) * self._cmd_backward_scale
        else:
            self.cmd[0] = 0.0

        # left-x for side moving left/right
        lx = -self.lx
        if lx > self.lin_vel_deadband:
            self.cmd[1] = self._cmd_y_min + (lx - self.lin_vel_deadband) * self._cmd_y_scale
        elif lx < -self.lin_vel_deadband:
            self.cmd[1] = -self._cmd_y_min + (lx + self.lin_vel_deadband) * self._cmd_y_scale
        else:
            self.cmd[1] = 0.0

        # right-x for turning left/right
        rx = -self.rx
        if rx > self.ang_vel_deadband:
            self.cmd[2] = self._cmd_yaw_min + (rx - self.ang_vel_deadband) * self._cmd_yaw_scale
        elif rx < -self.ang_vel_deadband:
            self.cmd[2] = -self._cmd_yaw_min + (rx + self.ang_vel_deadband) * self._cmd_yaw_scale
        else:
            self.cmd[2] = 0.0

    def _check_joint_limits(self):
        """检查关节位置是否超出安全范围"""
        for i in range(12):
            if self.qj[i] > self._pos_protect_high[i]:
                print(f"\033[91m[ERROR] Joint {i} position {self.qj[i]:.4f} exceeds {self._pos_protect_high[i]:.4f}\033[0m")
            elif self.qj[i] < self._pos_protect_low[i]:
                print(f"\033[91m[ERROR] Joint {i} position {self.qj[i]:.4f} below {self._pos_protect_low[i]:.4f}\033[0m")

    def _warn_by_joint_limits(self):
        """根据关节物理限位打印警告信息"""
        for i in range(12):
            # 当前位置接近上限，且还想往上走
            if self.qj[i] > self._pos_warn_high[i] and self.target_dof_pos[i] > self.qj[i] and self.target_dof_pos[i] > self._pos_protect_high[i]:
                print(f"\033[94m[WARNING] [JOINT_LIMIT] Joint {i}: {self.target_dof_pos[i]:.4f} > {self._pos_warn_high[i]:.4f}\033[0m")
            # 当前位置接近下限，且还想往下走
            elif self.qj[i] < self._pos_warn_low[i] and self.target_dof_pos[i] < self.qj[i] and self.target_dof_pos[i] < self._pos_protect_low[i]:
                print(f"\033[94m[WARNING] [JOINT_LIMIT] Joint {i}: {self.target_dof_pos[i]:.4f} < {self._pos_warn_low[i]:.4f}\033[0m")

    def _clip_by_torque_limit(self):
        """根据扭矩限制裁剪目标位置"""
        # tau = kp * (q_target - q_current) + kd * (0 - dq_current)
        kd_dq = (self.kds * self.dqj) / self.kps
        pos_limits_low = self.qj - self._torque_kp_ratio + kd_dq
        pos_limits_high = self.qj + self._torque_kp_ratio + kd_dq
        # np.clip(self.target_dof_pos, pos_limits_low, pos_limits_high, out=self.target_dof_pos)

        for i in range(12):
            if self.target_dof_pos[i] > pos_limits_high[i]:
                print(f"\033[93m[WARNING] [TORQUE] Joint {i} clipped (high): {self.target_dof_pos[i]:.4f} -> {pos_limits_high[i]:.4f}\033[0m")
                self.target_dof_pos[i] = pos_limits_high[i]
            elif self.target_dof_pos[i] < pos_limits_low[i]:
                print(f"\033[93m[WARNING] [TORQUE] Joint {i} clipped (low): {self.target_dof_pos[i]:.4f} -> {pos_limits_low[i]:.4f}\033[0m")
                self.target_dof_pos[i] = pos_limits_low[i]

    def run(self):
        """主控制循环：构建观测、推理策略动作并下发电机目标"""
        # ---- 1 读取最新低层状态 ----
        if self.latest_low_state is not None:
            self.low_state = self.latest_low_state

        # ---- 2 读取最新遥控器状态 ----
        if self.latest_wc is not None:
            wc = self.latest_wc
            self.keys = wc.keys
            self.lx, self.ly, self.rx = wc.lx, wc.ly, wc.rx

        # ---- 3 构造观测并推理动作 ----
        motor_state = self.low_state.motor_state
        joint2motor_idx = self.joint2motor_idx
        for i in range(12):
            motor_idx = joint2motor_idx[i]
            self.qj[i] = motor_state[motor_idx].q
            self.dqj[i] = motor_state[motor_idx].dq
        self._check_joint_limits()

        ang_vel = np.array([self.low_state.imu_state.gyroscope], dtype=np.float32) * self.obs_scales_ang_vel
        quat = self.low_state.imu_state.quaternion
        gravity_orientation = get_gravity_orientation(quat)  # IMU 四元数顺序 w x y z

        if self.use_remote_controller:
            self._process_remote_command()
        else:
            self.cmd.fill(0.0)

        # 就地计算
        np.subtract(self.qj, self.default_angles, out=self.qj_obs)
        self.qj_obs *= self.obs_scales_dof_pos
        np.multiply(self.dqj, self.obs_scales_dof_vel, out=self.dqj_obs)

        self.obs[:3] = ang_vel
        self.obs[3:6] = gravity_orientation
        self.obs[6:9] = self.cmd * self.command_scale
        self.obs[9:21] = self.qj_obs
        self.obs[21:33] = self.dqj_obs
        self.obs[33:45] = self.action
        np.clip(self.obs, -100.0, 100.0, out=self.obs)

        # 复用预分配的 Tensor
        self.obs_tensor[0] = torch.from_numpy(self.obs)
        # 构建不含 command 的观测用于历史
        self.obs_without_command[:6] = self.obs[:6]
        self.obs_without_command[6:] = self.obs[9:]
        # 滚动历史缓冲区
        self.proprio_history[:-1] = self.proprio_history[1:].copy()
        self.proprio_history[-1] = self.obs_without_command
        # 复用预分配的 history_tensor
        self.history_tensor[0] = torch.concat((torch.from_numpy(self.proprio_history.ravel()), self.obs_tensor[0, 6:9]))
        
        # 策略推理
        action_tensor = self.policy(self.obs_tensor, self.history_tensor)
        np.copyto(self.action, action_tensor[0].numpy())

        # 计算目标位置
        np.multiply(self.action, self.action_scale, out=self.target_dof_pos)
        self.target_dof_pos += self.default_angles
        self._warn_by_joint_limits()
        self._clip_by_torque_limit()

        motor_cmd = self.low_cmd.motor_cmd
        for i in range(12):
            motor_idx = joint2motor_idx[i]
            cmd = motor_cmd[motor_idx]
            cmd.q = self.target_dof_pos[i]
            cmd.dq = 0.0
            cmd.kp = 20.0
            cmd.kd = 0.5
            cmd.tau = 0.0
        self.send_cmd(self.low_cmd)

        # === 调试：遥控器 & 模型输出 ===
        # print(f"RC: lx={self.lx:+.2f} ly={self.ly:+.2f} rx={self.rx:+.2f}")
        # print(f"OBS cmd: {self.obs[6:9]}")  # 遥控器信号在 obs 的位置
        # print(f"RAW action: {self.action[:4]}...")  # 只看前 4 个，防止刷屏
        # print(f"TARGET Q: {target_dof_pos[::3]}")  # 每 3 个关节抽 1 个，易读


if __name__ == "__main__":
    config_path = f"{LEGGED_GYM_ROOT_DIR}/deploy/deploy_mujoco/configs/go2.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    control_dt = config["control_dt"]

    torch.set_grad_enabled(False)  # 全局禁用梯度计算
    torch.set_num_threads(1)  # 限制单线程推理以降低延迟波动

    # 构造控制器并依次执行安全启停流程
    controller = Controller(config)
    controller.zero_torque_state()
    controller.move_to_default_pos()
    controller.default_pos_state()

    while True:
        # 精确定时
        current_time = time.perf_counter()
        try:
            controller.run()
            if (controller.keys >> 3) & 1:
                break
            # 补偿计算耗时，保持稳定的控制周期
            elapsed = time.perf_counter() - current_time
            # print(f"Control loop time: {elapsed * 1000:.3f} ms")
            sleep_time = control_dt - elapsed - 0.0005  # 预留 0.5 ms 余量
            if sleep_time > 0:
                time.sleep(sleep_time)
            while (time.perf_counter() - current_time) < control_dt:
                pass
            print(f"Control loop time: {(time.perf_counter() - current_time) * 1000:.3f} ms")
        except KeyboardInterrupt:
            break

    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    print("Exit")
