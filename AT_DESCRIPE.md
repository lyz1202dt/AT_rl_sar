# Robot Lab训练机器人基本流程命令示例（GO2机器人）

RobotLab-Isaac-Velocity-Flat-ATDog-Dog_Arm-v0

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
  --resume \
  --load_run=2026-05-06_13-31-42 \
  --checkpoint=model_3098.pt \
  --headless 



```

## 导出模型：

```bash
cd /workspace/isaaclab_extension_template
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-ATDog-Dog-v0 \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog_flat/2026-05-10_13-02-45/model_4097.pt \
  --num_envs=10


/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Rough-ATDog-Dog2-v0 \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/atdog_dog2_rough/2026-05-08_13-33-53/model_13700.pt \
  --num_envs=10

```

## 复制导出的模型到宿主机

```bash
docker cp robot-lab:/workspace/isaaclab_extension_template/logs/rsl_rl/unitree_go2_flat/2026-04-19_08-25-46/exported/policy.pt  ./exported
```
