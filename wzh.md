一    检查温度和风扇转速cpu gpu
watch -n 0.3 sensors
watch -n 0.3 nvidia-smi

二    删除文件夹
cd 路径
rm -rf 文件夹名

三    创建文件夹
cd 路径
mkdir -p 文件夹名

四    copy文件夹到容器
docker cp robot-lab:/workspace/isaaclab_extension_template/logs/rsl_rl/unitree_go2_flat/2026-04-19_08-25-46/exported/policy.pt  ./exported
# 从容器复制到主机
docker cp <容器名>:<容器内路径> <主机路径>

# 从主机复制到容器
docker cp <主机路径> <容器名>:<容器内路径>

五    进入容器
docker exec -it robot-lab bash

六    容器内文件操作
# 创建文件夹
mkdir /workspace/isaaclab_extension_template/logs/test

# 创建多层目录
mkdir -p /workspace/isaaclab_extension_template/logs/test/subdir

# 删除空文件夹
rmdir /workspace/isaaclab_extension_template/logs/test

# 删除文件夹（包含内容）
rm -rf /workspace/isaaclab_extension_template/logs/test

# 删除文件
rm /workspace/isaaclab_extension_template/logs/test.txt

# 移动/重命名文件
mv /workspace/isaaclab_extension_template/logs/old.txt /workspace/isaaclab_extension_template/logs/new.txt

# 复制文件
cp /workspace/isaaclab_extension_template/logs/file.txt /workspace/isaaclab_extension_template/logs/backup/

# 查看文件内容
cat /workspace/isaaclab_extension_template/logs/file.txt

七    常用操作速查表
| 操作 | 宿主机命令 | 容器内命令 |
|------|----------|----------|
| 创建目录 | mkdir xxx | mkdir /workspace/xxx |
| 删除目录(含内容) | rm -rf xxx | rm -rf /workspace/xxx |
| 删除文件 | rm xxx | rm /workspace/xxx |
| 复制文件到容器 | docker cp xxx <容器>:/workspace/ | - |
| 从容器复制文件 | docker cp <容器>:/workspace/xxx . | - |

八    实用示例
# 查看容器内日志
docker exec robot-lab cat /workspace/isaaclab_extension_template/logs/train.log

# 备份容器内文件到宿主机
docker cp robot-lab:/workspace/isaaclab_extension_template/logs/model.pt ./backup/

# 导出训练好的模型
docker cp robot-lab:/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog2_flat/2026-04-26_01-52-12/exported/policy.pt ./exported/

# 将宿主机代码同步到容器
docker cp ./src/. robot-lab:/workspace/isaaclab_extension_template/src/

九    注意事项
# 路径对应关系
宿主机路径                              容器内路径
/home/zhangjiayi/RL/AT_rl_sar/logs  ←→  /workspace/isaaclab_extension_template/logs
./（项目根目录）                        ←→  /workspace/isaaclab_extension_template

# 时间戳目录：带 2026-04-26_01-52-12 格式的目录是训练自动生成的，直接删除即可
docker exec robot-lab rm -rf /workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog2_flat/2026-04-26_01-52-12

================================================================================
========================== 机器人训练开发指南 ====================================
================================================================================

十    项目介绍
# robot_lab 是基于 IsaacLab 的 RL 强化学习扩展库，用于机器人训练
# 支持多种机器人：四足机器人、人形机器人、轮式机器人
# 支持 RSL-RL、CusRL、Skrl 等强化学习算法

支持的机器人型号：
- 四足机器人：Unitree Go2, Anymal D, Unitree B2, Unitree A1, Deeprobotics Lite3 等
- 轮式机器人：Unitree Go2W, Unitree B2W, Deeprobotics M20 等
- 人形机器人：Unitree G1, Unitree H1, FFTAI GR1T1/GR1T2, Booster T1 等

十一   Docker 环境搭建
# 1. 构建 Docker 镜像
cd ~/RL/AT_rl_sar/docker
./container.sh build

# 2. 启动容器
./container.sh start

# 3. 进入容器
./container.sh enter

# 4. 停止容器
./container.sh stop

# 5. 重启容器
./container.sh restart

十二   开始训练（RSL-RL 算法）
# 进入容器后，切换到工作目录
cd /workspace/isaaclab_extension_template

# 训练四足机器人（Unitree Go2）
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --num_envs 2048 \
  --headless \
  --max_iterations 3000

# 训练四足机器人（AT_Dog2）
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-ATDog-Dog2-v0 \
  --num_envs 2048 \
  --headless \
  --max_iterations 3000

# 训练人形机器人（Unitree G1）
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-Unitree-G1-v0 \
  --num_envs 2048 \
  --headless \
  --max_iterations 3000

# 继续之前的训练（添加 checkpoint 参数）
--checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/temp/model_1999.pt

十三   训练参数说明
--task=<ENV_NAME>        # 环境名称，决定训练哪个机器人
--num_envs=<NUM>         # 并行环境数量，2048 是常用值，GPU 显存不够可减少
--headless                # 无 GUI 模式，不显示仿真界面
--max_iterations=<NUM>    # 最大训练迭代次数
--checkpoint=<PATH>       # 继续训练时的模型路径（可选）

十四   常用环境名称（Task ID）
# 四足机器人
RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0
RobotLab-Isaac-Velocity-Flat-Anymal-D-v0
RobotLab-Isaac-Velocity-Flat-Unitree-B2-v0
RobotLab-Isaac-Velocity-Flat-ATDog-Dog2-v0

# 人形机器人
RobotLab-Isaac-Velocity-Flat-Unitree-G1-v0
RobotLab-Isaac-Velocity-Flat-Unitree-H1-v0
RobotLab-Isaac-Velocity-Flat-FFTAI-GR1T1-v0
RobotLab-Isaac-Velocity-Flat-Booster-T1-v0

# 轮式机器人
RobotLab-Isaac-Velocity-Flat-Unitree-Go2W-v0

十五   导出训练好的模型
# 导出 Unitree Go2 模型
cd /workspace/isaaclab_extension_template
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/unitree_go2_flat/2026-04-19_08-25-46/model_1999.pt \
  --num_envs=1 \
  --headless

# 导出 ATDog2 模型
cd /workspace/isaaclab_extension_template
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-ATDog-Dog2-v0 \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog2_flat/2026-04-25_13-03-08/model_1199.pt \
  --num_envs=1 \
  --headless

十六   将模型复制到宿主机
# 复制 Go2 模型
docker cp robot-lab:/workspace/isaaclab_extension_template/logs/rsl_rl/unitree_go2_flat/2026-04-19_08-25-46/exported/policy.pt ./exported/

# 复制 ATDog2 模型
docker cp robot-lab:/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog2_flat/2026-04-25_13-03-08/exported/policy.pt ./exported/

十七   训练结果目录结构
/workspace/isaaclab_extension_template/logs/rsl_rl/
├── atdog_dog2_flat/                    # 机器人类型
│   ├── 2026-04-25_13-03-08/          # 时间戳目录（训练自动生成）
│   │   ├── model_0000.pt              # 训练过程中的模型
│   │   ├── model_1199.pt              # 最后一个模型
│   │   └── exported/                  # 导出的模型
│   │       └── policy.pt              # 最终导出的策略模型
│   └── ...
└── unitree_go2_flat/
    └── ...

十八   删除旧的训练记录
# 删除特定时间戳的训练记录
docker exec robot-lab rm -rf /workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog2_flat/2026-04-26_01-52-12

# 删除所有带时间戳的目录
docker exec robot-lab rm -rf /workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog2_flat/2026-*

十九   使用其他强化学习算法
# CusRL 算法（实验性）
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/cusrl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --num_envs 2048 \
  --headless

# Skrl 算法
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --num_envs 2048 \
  --headless

二十   查看所有可用环境
# 在容器内执行
python scripts/tools/list_envs.py

二十一  项目代码结构
/workspace/isaaclab_extension_template/
├── source/robot_lab/              # 机器人扩展库源码
│   ├── robot_lab/
│   │   ├── assets/               # 机器人资产定义
│   │   │   ├── unitree.py       # Unitree 机器人
│   │   │   ├── atdog.py         # ATDog 机器人
│   │   │   └── ...
│   │   └── tasks/
│   │       ├── manager_based/   # 基于管理器的方式
│   │       │   └── locomotion/   #  locomotion 任务
│   │       │       └── velocity/ # 速度控制任务
│   │       └── direct/          # 直接控制方式
│   └── data/Robots/             # 机器人 URDF 和 mesh 文件
│       ├── atdog/dog2/          # ATDog 机器人数据
│       ├── unitree/             # Unitree 机器人数据
│       └── ...
├── scripts/
│   ├── reinforcement_learning/   # 强化学习训练脚本
│   │   ├── rsl_rl/             # RSL-RL 算法
│   │   │   ├── train.py        # 训练脚本
│   │   │   └── play.py         # 导出/播放脚本
│   │   ├── cusrl/              # CusRL 算法
│   │   └── skrl/               # Skrl 算法
│   └── tools/                   # 工具脚本
└── logs/                        # 训练日志和模型输出

二十二  添加新机器人的基本步骤
1. 准备机器人 URDF 文件，放入 source/robot_lab/data/Robots/<robot_name>/
2. 在 source/robot_lab/robot_lab/assets/ 中添加机器人定义
3. 在 source/robot_lab/robot_lab/tasks/ 中添加任务配置
4. 在 config/extension.toml 中注册新环境
5. 重新构建 Docker 镜像
6. 测试新环境

二十三  常见问题解决
# 问题1：GPU 内存不足
解决：减少 --num_envs 数量，如从 2048 降到 1024 或 512

# 问题2：容器无法启动
解决：重启 Docker 服务
sudo systemctl restart docker

# 问题3：训练中断，想继续训练
解决：使用 --checkpoint 参数指定之前的模型路径

# 问题4：PhysX GPU 错误
解决：可能是 GPU 驱动问题，尝试在宿主机执行
sudo rmmod nvidia
sudo modprobe nvidia
然后重启容器

二十四  训练不同目标的配置指南

## 工程目录结构

```
├── scripts/
│   ├── reinforcement_learning/   # 强化学习训练脚本
│   │   ├── rsl_rl/             # RSL-RL 算法
│   │   │   ├── train.py        # 训练脚本
│   │   │   └── play.py         # 导出/播放脚本
│   │   ├── cusrl/              # CusRL 算法
│   │   └── skrl/               # Skrl 算法
│   └── tools/                   # 工具脚本
├── source/robot_lab/
│   ├── robot_lab/
│   │   ├── assets/               # 机器人资产定义
│   │   └── tasks/
│   │       └── manager_based/   # 基于管理器的任务
│   │           └── locomotion/   # 运动任务
│   │               └── velocity/ # 速度控制任务
│   └── data/Robots/             # 机器人 URDF 和 mesh 文件
└── logs/                        # 训练日志和模型输出
```

## 训练不同目标需要修改的文件

### 1. 切换机器人类型

**修改方式**：通过命令行参数 `--task` 指定不同的环境名称

**环境名称格式**：
- 四足机器人：`RobotLab-Isaac-Velocity-Flat-<Robot>-v0`
- 人形机器人：`RobotLab-Isaac-Velocity-Flat-<Robot>-v0`
- 轮式机器人：`RobotLab-Isaac-Velocity-Flat-<Robot>-v0`

**示例**：
- 训练 Unitree Go2：`--task=RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0`
- 训练 ATDog：`--task=RobotLab-Isaac-Velocity-Flat-ATDog-Dog2-v0`
- 训练 Unitree G1：`--task=RobotLab-Isaac-Velocity-Flat-Unitree-G1-v0`

### 2. 修改地形类型

**修改方式**：选择不同的环境配置文件

**地形类型**：
- 平坦地形：使用 `flat_env_cfg.py`
- 粗糙地形：使用 `rough_env_cfg.py`

**环境名称对应**：
- 平坦地形：`RobotLab-Isaac-Velocity-Flat-<Robot>-v0`
- 粗糙地形：`RobotLab-Isaac-Velocity-Rough-<Robot>-v0`

### 3. 修改任务参数

**配置文件位置**：`source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/<robot_type>/<robot_name>/`

**主要修改项**：

#### 3.1 奖励函数调整
在 `rough_env_cfg.py` 文件中修改奖励权重：
- 速度跟踪奖励：`self.rewards.track_lin_vel_xy_exp.weight`
- 姿态惩罚：`self.rewards.flat_orientation_l2.weight`
- 关节惩罚：`self.rewards.joint_torques_l2.weight`
- 接触力惩罚：`self.rewards.contact_forces.weight`

#### 3.2 命令范围调整
在 `rough_env_cfg.py` 文件中修改速度命令范围：
```python
self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)  # 前进/后退速度范围
self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)   # 左右平移速度范围
self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)   # 转向速度范围
```

#### 3.3 观察空间调整
在 `rough_env_cfg.py` 文件中修改观察参数：
- 基础速度：`self.observations.policy.base_lin_vel.scale`
- 关节位置：`self.observations.policy.joint_pos.scale`
- 关节速度：`self.observations.policy.joint_vel.scale`

#### 3.4 动作空间调整
在 `rough_env_cfg.py` 文件中修改动作参数：
- 关节位置缩放：`self.actions.joint_pos.scale`
- 关节位置限制：`self.actions.joint_pos.clip`

### 4. 修改训练参数

**通过命令行参数修改**：
- 并行环境数量：`--num_envs 2048`
- 训练迭代次数：`--max_iterations 3000`
- 继续训练：`--checkpoint=<path_to_model>`
- 无头模式：`--headless`

### 5. 更换强化学习算法

**修改方式**：使用不同的训练脚本

**支持的算法**：
- RSL-RL：`scripts/reinforcement_learning/rsl_rl/train.py`
- CusRL：`scripts/reinforcement_learning/cusrl/train.py`
- Skrl：`scripts/reinforcement_learning/skrl/train.py`

## 训练流程示例

1. **训练四足机器人（Unitree Go2）**：
   ```bash
   /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
     --task=RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 \
     --num_envs 2048 \
     --headless \
     --max_iterations 3000
   ```

2. **训练人形机器人（Unitree G1）**：
   ```bash
   /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
     --task=RobotLab-Isaac-Velocity-Flat-Unitree-G1-v0 \
     --num_envs 2048 \
     --headless \
     --max_iterations 3000
   ```

3. **继续训练**：
   ```bash
   /workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
     --task=RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 \
     --num_envs 2048 \
     --headless \
     --max_iterations 3000 \
     --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/unitree_go2_flat/2026-04-19_08-25-46/model_1999.pt
   ```

## 训练结果目录

训练结果存储在 `logs/rsl_rl/<robot_name>_flat/` 目录下，每个训练会生成一个时间戳目录，包含训练过程中的模型和导出的策略模型。

## 总结

要训练不同目标，主要需要修改以下几个方面：
1. **机器人类型**：通过 `--task` 参数选择不同的机器人环境
2. **地形类型**：选择 `Flat` 或 `Rough` 环境
3. **任务参数**：修改对应的配置文件中的奖励函数、命令范围、观察空间和动作空间
4. **训练参数**：通过命令行参数调整训练设置
5. **强化学习算法**：选择不同的训练脚本

通过合理配置这些参数，可以针对不同的机器人和训练目标进行定制化训练。


velocity_env_cfg.py (定义奖励函数注册)
        ↓
  RewardsCfg 类中定义 RewTerm
        ↓
rough_env_cfg.py (设置奖励权重)
        ↓
   self.rewards.xxx.weight = 值


   rewards.py 中的函数 → 计算奖励值 → 乘以 weight → 最终奖励


#  参数检测脚本
   python3 /home/zhangjiayi/RL/AT_rl_sar/scripts/tools/system_monitor.py

#  监控训练
   python3 scripts/tools/monitor_training_live.py --logdir logs/rsl_rl/atdog_dog2_flat/2026-04-28_12-53-05

  
# 查看文件是否存在
ls -la /tmp/training_plot.png

# 用图片查看器打开
xdg-open /tmp/training_plot.png  # Linux
eog /tmp/training_plot.png       # GNOME
feh /tmp/training_plot.png       # 轻量级查看器