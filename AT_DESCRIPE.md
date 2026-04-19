# Robot Lab训练机器人基本流程命令示例（GO2机器人）

## 创建环境：

```bash
cd /home/lyz/Project/robot_lab-2.3.0/docker
docker compose --env-file .env.base --file docker-compose.yaml build --no-cache robot-lab
docker compose --env-file .env.base --file docker-compose.yaml up -d
docker exec -it robot-lab /bin/bash
```

## 开始训练

```bash
cd /workspace/isaaclab_extension_template
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 --num_envs 2048 --headless

```

## 导出模型：

```bash
cd /workspace/isaaclab_extension_template
/workspace/isaaclab/isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Flat-Unitree-Go2-v0 \
  --checkpoint=/workspace/isaaclab_extension_template/logs/rsl_rl/unitree_go2_flat/2026-04-19_08-25-46/model_1999.pt \
  --num_envs=1 \
  --headless

```

## 复制导出的模型到宿主机


```bash
docker cp robot-lab:/workspace/isaaclab_extension_template/logs/rsl_rl/unitree_go2_flat/2026-04-19_08-25-46/exported/policy.pt  ./exported
```
