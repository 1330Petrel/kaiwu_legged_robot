# kaiwu_legged_robot

2025 腾讯开悟具身智能强化学习运动控制赛道决赛参赛代码仓库

## 项目框架

```plaintext
.
├─conf/                  # 开悟平台配置项
├─environment/           # 仿真环境与地形
├─Baseline/              # 基线实现与改进
├─SLR/                   # SLR 论文复现
├─CTS/                   # CTS 论文复现
├─deploy/                # 仿真与实机部署
└─report.pdf             # 技术报告
```

- 环境层：开悟平台提供的仿真环境 [environment/legged_robot.py](environment/legged_robot.py) 和地形实现 [environment/terrain.py](environment/terrain.py)。
- 训练层：Baseline/SLR/CTS 服从开悟强化学习训练框架结构，含 `agent.py`、`algorithm/algorithm.py`、`feature/definition.py`、`feature/reward_process.py`、`model/model.py`、`workflow/train_workflow.py`、`conf/`。
    1. Baseline：基于开悟平台提供的基线进行改进；
    2. SLR：[SLR: Learning Quadruped Locomotion without Privileged Information](https://11chens.github.io/SLR/) 论文复现；
    3. CTS：[CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion](https://clearlab-sustech.github.io/concurrentTS/) 论文复现。
- 部署层：仿真/实机部署在 [deploy/deploy_mujoco](deploy/deploy_mujoco) 与 [deploy/deploy_real](deploy/deploy_real)，预训练权重在 [deploy/pre_train/go2](deploy/pre_train/go2)。

### CTS 文件框架

```plaintext
CTS/
├─agent.py               # 接口实现
├─algorithm/
│  └─algorithm.py        # PPO/GAE 等算法实现
├─conf/
│  ├─conf.py             # 强化学习超参
│  └─train_env_conf.toml # 环境/任务超参
├─feature/
│  ├─definition.py       # Rollout 经验池与特征构建
│  └─reward_process.py   # 自定义奖励函数与 hack
├─model/
│  └─model.py            # 策略/价值网络
└─workflow/
    └─train_workflow.py   # 训练循环实现
```

### deploy 文件框架

```plaintext
deploy/
├─deploy_mujoco/
│  ├─deploy_mujoco_go2.py   # Mujoco 仿真加载策略入口
│  ├─new.py                 # 新版本仿真脚本入口
│  ├─scene_terrain.xml      # 仿真地形文件
│  ├─unitree_mujoco.py      # 仿真脚本
│  ├─common/
│  │  ├─command_helper.py
│  │  ├─remote_controller.py
│  │  └─rotation_helper.py
│  └─configs/
│     └─go2.yaml            # 仿真部署配置
├─deploy_real/
│  ├─deploy_real_go2.py     # 实机部署入口
│  ├─new.py                 # 新版本实机脚本入口
│  ├─common/
│  │  ├─command_helper.py
│  │  ├─remote_controller.py
│  │  └─rotation_helper.py
│  └─configs/
│     └─go2.yaml            # 实机部署配置
└─pre_train/
    └─go2/                  # 预训练策略权重
```

## 部署

- 依赖：[unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym), [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python), [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)

- 将 `deploy` 目录替换 `unitree_rl_gym` 原有的 `deploy` 目录

### Sim2Sim

- 复制仿真脚本到 `unitree_mujoco/simulate_python` 目录

    ```bash
    cp deploy/deploy_mujoco/unitree_mujoco.py unitree_mujoco/simulate_python
    ```

- 复制地形文件到 `unitree_mujoco/unitree_robots/go2` 目录

    ```bash
    cp deploy/deploy_mujoco/scene_terrain.xml unitree_mujoco/unitree_robots/go2/scene_terrain.xml
    ```

- 修改 `unitree_mujoco/simulate_python/config.py` 中的 `ROBOT_SCENE` 字段: `ROBOT_SCENE = "../unitree_robots/" + ROBOT + "/scene_terrain.xml"`

- 运行 mujoco

    ```bash
    cd unitree_mujoco/simulate_python
    python3 unitree_mujoco_1.py
    ```

- 运行仿真

    ```bash
    cd unitree_rl_gym/deploy/deploy_mujoco
    python3 deploy_mujoco_go2.py
    ```

### Sim2Real

- 参考 [部署说明](https://aiarena.tencent.com/docs/p-competition-legged_robot_locomotion_control/70.2.1/guidebook/dev-guide/deployment/#%E9%83%A8%E7%BD%B2%E6%96%B9%E5%BC%8F%E4%B8%80%E6%8E%A8%E8%8D%90)

- 实机部署

    ```bash
    cd unitree_rl_gym/deploy/deploy_real
    python3 deploy_real_go2.py --<robot_ip>
    ```
