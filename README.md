# kaiwu_legged_robot

## 部署

- 依赖：[unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym), [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python), [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)

- 将 `deploy` 目录替换 `unitree_rl_gym` 原有的 `deploy` 目录

- 预训练模型位于 `deploy/pre_train/go2/` 目录下

### Sim2Sim

- 复制仿真脚本

    ```bash
    cp unitree_mujoco_1.py unitree_mujoco/simulate_python
    ```

- 复制地形文件

    ```bash
    cp scene_terrain_1.xml unitree_mujoco/unitree_robots/go2
    ```

- 修改 `unitree_mujoco/simulate_python/config.py` 中的 `ROBOT_SCENE` 字段: `ROBOT_SCENE = "../unitree_robots/" + ROBOT + "/scene_terrain_1.xml"`

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

> [!IMPORTANT]
>
> 修改 `deploy/deploy_mujoco/configs/go2.yaml` 中的 `policy_path: "{LEGGED_GYM_ROOT_DIR}/deploy/pre_train/go2/V2-4-1_83357.pt"` 为需要的模型路径

### Sim2Real

- 参考 [部署说明](https://aiarena.tencent.com/docs/p-competition-legged_robot_locomotion_control/70.2.1/guidebook/dev-guide/deployment/#%E9%83%A8%E7%BD%B2%E6%96%B9%E5%BC%8F%E4%B8%80%E6%8E%A8%E8%8D%90)

- 实机部署

    ```bash
    cd unitree_rl_gym/deploy/deploy_real
    python3 deploy_real_go2.py
    ```

> [!IMPORTANT]
>
> 修改 `deploy/deploy_real/configs/go2.yaml` 中的 `policy_path: "{LEGGED_GYM_ROOT_DIR}/deploy/pre_train/go2/V2-4-1_83357.pt"` 为需要的模型路径
