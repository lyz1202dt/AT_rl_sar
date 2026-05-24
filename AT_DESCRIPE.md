# Robot Lab训练机器人基本流程命令示例（GO2机器人）

export http_proxy=192.168.2.180:7890
export https_proxy=192.168.2.180:7890
export socket_proxy=192.168.2.180:7890

ssh -p 3022  sw@shenweitechnology.com

## 创建环境：

```bash
docker/container.sh build
docker/container.sh start
docker/container.sh enter
```

## 开始训练

```bash
cd /workspace/isaaclab_extension_template
/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_extension_template/scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-ATDog-Dog-v0 \
  --num_envs=8000 \
  --max_iterations=1000\
  --headless 

  --resume \
  --load_run=2026-05-06_13-31-42 \
  --checkpoint=model_3098.pt \


/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_extension_template/scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-ATDog-Dog_Arm-v0 \
  --num_envs=30000 \
  --max_iterations=200\
  --headless


```

## 导出模型：

```bash
cd /workspace/isaaclab_extension_template
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-ATDog-Dog_Arm-v0 \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_arm_flat/2026-05-23_14-26-00/model_299.pt \
  --num_envs=1\
  --headless


/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Rough-ATDog-Dog2-v0 \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog2_rough/2026-05-08_13-33-53/model_13700.pt \
  --num_envs=10

```

## 复制导出的模型到宿主机

初始：2026-05-23_14-26-00

scp -P 3022 sw@shenweitechnology.com:/home/sw/code/AT_rl_sar/logs/rsl_rl/atdog_arm_flat/2026-05-23_14-26-00/exported/policy.onnx ~/桌面/

```bash
docker cp robot-lab:/workspace/isaaclab_extension_template/logs/rsl_rl/unitree_go2_flat/2026-04-19_08-25-46/exported/policy.pt  ./exported
```



