# Robot Lab训练机器人基本流程命令示例（GO2机器人）


ATDOG2
任务名称：
RobotLab-Isaac-Velocity-Flat-ATDog-Dog2-v0      --平地
RobotLab-Isaac-Velocity-Rough-ATDog-Dog2-v0     --崎岖
RobotLab-Isaac-Velocity-Stairs-ATDog-Dog2-v0    --台阶
RobotLab-Isaac-Velocity-Sand-ATDog-Dog2-v0      --沙地
RobotLab-Isaac-Velocity-Slope-ATDog-Dog2-v0     --斜坡

ATDOG2ARM
任务名称：
RobotLab-Isaac-Velocity-Rough-ATDog-Dog2-Arm-v0   --任务赛（减速带）

ATDOG3
任务名称：
RobotLab-Isaac-Velocity-Flat-ATDog-Dog3-v0      --平地
RobotLab-Isaac-Velocity-Rough-ATDog-Dog3-v0     --崎岖
RobotLab-Isaac-Velocity-Stairs-ATDog-Dog3-v0    --台阶
RobotLab-Isaac-Velocity-Sand-ATDog-Dog3-v0      --沙地
RobotLab-Isaac-Velocity-Slope-ATDog-Dog3-v0     --斜坡

ATDOG3ARM
任务名称：
RobotLab-Isaac-Velocity-Rough-ATDog-Dog3-Arm-v0   --任务赛（减速带）




注意：ATDOG已经废弃，请使用ATDOG3


export http_proxy=192.168.2.180:7890
export https_proxy=192.168.2.180:7890
export socket_proxy=192.168.2.180:7890

1.  ssh -p 3022  sw@shenweitechnology.com

2.  cd /home/sw/code/AT_rl_sar

3.  
export http_proxy=192.168.2.180:7890
export https_proxy=192.168.2.180:7890

4.  docker/container.sh start

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

cd /workspace/isaaclab_extension_template
/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_extension_template/scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-ATDog-Dog2-Arm-v0 \
  --num_envs=10000 \
  --max_iterations=1000\
  --headless \
  --resume \
  --load_run=2026-06-10_01-47-29 \
  --checkpoint=model_35988.pt 


/workspace/isaaclab/isaaclab.sh -p /workspace/isaaclab_extension_template/scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Flat-ATDog-Dog_Arm-v0 \
  --num_envs=10000 \
  --max_iterations=2000\
  --headless



```

## 导出模型：

```bash
cd /workspace/isaaclab_extension_template
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-ATDog-Dog_Arm-v0 \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_arm_flat/2026-05-31_14-32-45/model_4999.pt \
  --num_envs=10\
  --headless

/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Rough-ATDog-Dog2-Arm-v0 \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog2_arm_rough/2026-06-10_01-47-29/model_35988.pt \
  --num_envs=1\
  --headless



/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-ATDog-Dog2-v0 \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog_flat/2026-05-31_14-32-45/model_4999.pt\
  --num_envs=10

```

## 复制导出的模型到宿主机

初始：2026-05-23_14-26-00

scp -P 3022 sw@shenweitechnology.com:/home/sw/code/AT_rl_sar/logs/rsl_rl/atdog_dog2_flat/2026-06-07_18-30-45/exported/policy.pt ~/桌面/

scp -P 3022 sw@shenweitechnology.com:/home/sw/code/AT_rl_sar/logs/rsl_rl/atdog_dog2_arm_rough/2026-06-10_01-47-29/exported/policy.pt ~/桌面/

```bash
docker cp robot-lab:/workspace/isaaclab_extension_template/logs/rsl_rl/unitree_go2_flat/2026-04-19_08-25-46/exported/policy.pt  ./exported
```



