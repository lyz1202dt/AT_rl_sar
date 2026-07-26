# HIMLoco -> AT_rl_sar 移植记录

## 1. 目标与范围

本文记录将 `HIMLoco` 算法接入 `AT_rl_sar` 的完整过程，目标是：

- 不修改 docker、镜像和外部 `IsaacLab` 源码。
- 在 `AT_rl_sar` 仓内本地落一套 `HIM` 算法实现。
- 兼容 `Isaac Lab + Gymnasium + rsl_rl` 的训练、回放和导出链路。
- 让 `ATDog` 和标准四足任务都能通过 `rsl_rl_him_cfg_entry_point` 启动训练。
- 训练日志尽量贴近现有 PPO 输出，并补齐速度误差指标。

这次移植最终采用的是：

- 算法层面保留 `HIMLoco` 的核心结构。
- 训练配置、任务注册、场景组织、观测管理沿用 `AT_rl_sar` 现有体系。
- 所有适配都在本仓完成，不依赖外部改 `rsl-rl-lib`。


## 2. 最终落地结构

### 2.1 本地 HIM 模块

新增目录：

- `source/robot_lab/robot_lab/him/`

包含文件：

- `him_actor_critic.py`
- `him_estimator.py`
- `him_ppo.py`
- `him_rollout_storage.py`
- `him_on_policy_runner.py`
- `him_vec_env_wrapper.py`
- `exporter.py`

这套实现负责把 `HIMLoco` 的策略网络、估计器、PPO 更新、rollout 存储、runner 和导出链路全部本地化。

### 2.2 训练 / 回放入口

已修改：

- `scripts/reinforcement_learning/rsl_rl/train.py`
- `scripts/reinforcement_learning/rsl_rl/play.py`
- `scripts/reinforcement_learning/rsl_rl/play_cs.py`

入口侧新增了 `HIMOnPolicyRunner` 分支，并在需要时自动使用 `HIMVecEnvWrapper` 包装环境。

### 2.3 环境与任务侧

关键环境配置入口：

- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/velocity_env_cfg.py`
- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/commands.py`

任务注册和配置已补到：

- `quadruped/unitree_a1`
- `quadruped/atdog/dog2`
- `quadruped/atdog/dog2_arm`
- `quadruped/atdog/dog3`
- `quadruped/atdog/dog3_arm`
- `quadruped/atdog/dog4`
- `wheeled/atdog`
- `wheeled/atdog_arm`


## 3. 为什么这样接

这次不是把 `AT_rl_sar` 移植到 `HIMLoco`，而是把 `HIMLoco` 算法接到 `AT_rl_sar` 训练框架里。原因很直接：

- `AT_rl_sar` 已经沉淀了任务注册、场景配置、terrain、reward、观测管理和 Isaac Lab 适配。
- `HIMLoco` 的核心增量主要在算法层，不在任务框架层。
- 如果反过来把 `AT_rl_sar` 的大量任务和场景系统搬到 `HIMLoco`，改动面会更大，回归风险也更高。

因此本次策略是：

1. 固定 `AT_rl_sar` 的任务系统。
2. 在本仓补本地 `HIM` 算法栈。
3. 用最小侵入方式接到现有 `rsl_rl` 入口。


## 4. 核心适配点

### 4.1 Runner 接线

`train.py` 原本只识别：

- `OnPolicyRunner`
- `DistillationRunner`

现在新增：

- `HIMOnPolicyRunner`

逻辑是：

- `agent_cfg.class_name == "HIMOnPolicyRunner"` 时使用 `HIMVecEnvWrapper`
- 否则继续走原有 `RslRlVecEnvWrapper`

这样不会影响现有 PPO 和蒸馏链路。

### 4.2 Isaac Lab / Gymnasium 接口适配

`HIMLoco` 原始实现默认面向较旧的 legged gym 风格接口，直接接 `Isaac Lab` 会出现多类问题：

- `env.reset()` / `env.step()` 返回签名不同
- 外层可能是 `OrderEnforcing` wrapper
- `RslRlVecEnvWrapper` 不直接暴露 `num_privileged_obs`
- 观测是按 `policy` / `critic` group 组织，而不是单个拼好的 tensor

为此新增了 `HIMVecEnvWrapper`，负责：

- 包装 `RslRlVecEnvWrapper`
- 从 `policy` group 提取 actor obs
- 从 `critic` group 提取 privileged obs
- 缓存最近一次 actor / critic obs
- 提供 `get_observations()`、`get_privileged_observations()`、`get_critic_observations()`
- 兼容 `reset()` / `step()` 的 Isaac Lab 返回格式

这个 wrapper 是整次移植里最关键的一层。

### 4.3 观测契约适配

`HIMLoco` 对观测有两个硬约束：

- actor 输入是 history obs
- estimator / critic 依赖 privileged obs

而 `AT_rl_sar` 的观测是由 observation manager 按 term 组织的，所以不能再硬编码维度和切片。现在的处理方式是：

- 从 `policy` group 自动推导 one-step obs 维度
- 将 Isaac Lab 的 term-major 历史布局重排为 HIM 使用的 current-frame-first 布局
- 从 `critic` group 自动推导 estimator velocity slice
- 从 policy / critic 的共享 term 自动推导 estimator target slice

对应能力都落在：

- `source/robot_lab/robot_lab/him/him_vec_env_wrapper.py`

这解决了最开始的两个典型报错：

- `AttributeError: 'RslRlVecEnvWrapper' object has no attribute 'num_privileged_obs'`
- `ValueError: Actor observation dim 256 is not divisible by one-step dim 45`

后者的根因是原始 `HIMLoco` 把 `45` 当成固定 one-step 输入，但当前任务实际 policy obs 已经开启 history，actor 输入维度不再是写死值。

### 4.4 Estimator 切片不再写死

`HIMLoco` 原版默认：

- `next_critic_obs[:, 45:48]` 是 `base_lin_vel`
- `next_critic_obs[:, 3:48]` 是 `target input`

这在 `AT_rl_sar` 里并不可靠，因为 observation term 排布由环境配置决定。现在统一以 observation manager 的实际布局为准：

- `num_one_step_obs` 从 policy observation terms 自动推导
- `estimator_vel_slice` 从 critic 的 `base_lin_vel` term 自动推导
- `estimator_target_slice` 从 policy / critic 共享且连续的 terms 自动推导
- 即使旧配置中的切片维度合法，只要和环境布局不一致，也优先使用环境推导结果

对应逻辑在：

- `source/robot_lab/robot_lab/him/him_on_policy_runner.py`

这样可以避免配置中的 `(45, 48)` 在当前 critic 布局中误指向 action，而不是 `base_lin_vel`。

### 4.5 对 done 样本做 mask

原始 `HIMLoco` 某些实现会依赖环境返回额外的 termination privileged obs。为了减少对环境接口的侵入，这次采用的是更保守的方案：

- estimator 更新时直接对 `done` 样本做 mask
- 不强依赖环境额外返回 `termination_privileged_obs`

这样更适合 `AT_rl_sar` 当前环境栈。

### 4.6 导出链路修复

`HIMLoco` 原始导出逻辑常把输入维度写死成 `45`，这在当前仓里会直接卡住部署。现在本地 exporter 已改成按当前 actor-critic 实例的真实维度导出：

- `num_one_step_obs` 来自模型实例
- ONNX dummy input 使用 `actor_critic.num_actor_obs`

文件：

- `source/robot_lab/robot_lab/him/exporter.py`

同时 `play.py` / `play_cs.py` 已支持对 `HIMOnPolicyRunner` 正常加载和导出。


## 5. 配置注册方式

### 5.1 配置文件组织原则

`HIM` 配置现在按本仓 PPO 配置风格显式展开，不再依赖公共 `_him_cfg.py` 模板。这样做是为了：

- 更接近现有 `cusrl_ppo_cfg.py` / `rsl_rl_ppo_cfg.py`
- 每个任务的超参更直观
- 后续局部改动不会牵一发动全身

例如：

- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/config/quadruped/atdog/dog3/agents/rsl_rl_him_cfg.py`

### 5.2 注册入口

每个任务的 `__init__.py` 都补了：

- `rsl_rl_him_cfg_entry_point`

这样 Hydra 才能识别：

```bash
--agent=rsl_rl_him_cfg_entry_point
```

这个问题对应修复了下面的报错：

```text
ValueError: Could not find configuration for the environment ...
Please check that the gym registry has the entry point: 'rsl_rl_him_cfg_entry_point'
```


## 6. 训练日志适配

### 6.1 目标

用户希望 `HIM` 的终端训练反馈尽量接近现有 PPO，例如要能看到：

- `Mean reward`
- `Mean episode length`
- `Episode_Reward/*`
- `Episode_Termination/*`
- `Metrics/base_velocity/*`

并且这两项必须优先展示：

- `Metrics/base_velocity/error_vel_xy`
- `Metrics/base_velocity/error_vel_yaw`

### 6.2 实现

`HIMOnPolicyRunner` 现在会：

- 聚合 `infos["log"]`
- 读取 Isaac Lab manager 自动写回的 reward / metrics / termination 信息
- 缓存 `reset()` 阶段的 log，避免前几轮完全没有 metrics
- 当本轮没有完整 episode 结束时，回退到当前 partial trajectory 的平均 reward 和长度
- 打印更接近 PPO 的大块终端日志

另外，`HIMPPO.update()` 现在还会回传：

- `mean_entropy_loss`

因此日志中可同时看到：

- `Mean estimation loss`
- `Mean swap loss`
- `Mean entropy loss`

### 6.3 为什么之前只有两个速度误差

不是 runner 少打了，而是 Isaac Lab 原始 `UniformVelocityCommand` 默认只统计：

- `error_vel_xy`
- `error_vel_yaw`

所以即便 runner 能打印，也没有更多字段可打。

### 6.4 本仓补充的速度误差指标

由于不能改外部 `IsaacLab` 源码，因此在本仓本地 command 扩展了 metric。修改文件：

- `source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/commands.py`

本地 `UniformThresholdVelocityCommand` 现在额外输出：

- `Metrics/base_velocity/error_vel_x`
- `Metrics/base_velocity/error_vel_y`
- `Metrics/base_velocity/error_vel_z`
- `Metrics/base_velocity/error_vel_dir`

含义如下：

- `error_vel_x`: base frame 下 x 方向线速度误差
- `error_vel_y`: base frame 下 y 方向线速度误差
- `error_vel_z`: base z 方向线速度绝对值，等价于相对目标 `0` 的误差
- `error_vel_dir`: 实际 XY 速度方向与命令 XY 速度方向的夹角误差

同时 runner 里也把这些 metric 调整到优先输出顺序，因此现在终端能先看到细粒度速度误差，再看到其他 reward 和 termination 指标。


## 7. 这次实际遇到的问题与修复

### 7.1 `ModuleNotFoundError: No module named 'isaaclab'`

原因：

- 直接用系统 `python` 启动，没进 Isaac Lab 对应 Python 环境。

处理：

- 需要在 Isaac Lab 对应环境中运行，或者走 Isaac Lab 提供的启动脚本。

### 7.2 启动时被系统 `Killed`

现象：

- scene 创建完成后进仿真阶段被直接杀掉。

原因：

- 常见是显存或内存不足，尤其是 rough terrain + `4096` env。

处理建议：

- 先把 `num_envs` 降到 `64` 或 `128`
- 回放时关闭不必要 terrain curriculum
- 验证链路先用标准四足 rough 任务

### 7.3 `num_privileged_obs` / `get_privileged_observations` 缺失

原因：

- `RslRlVecEnvWrapper` 并不是按 `HIMLoco` 预期设计的。

处理：

- 在 `HIMVecEnvWrapper` 中统一适配，并从 observation groups 中提取 `policy` / `critic` 观测。

### 7.4 `HIMActorCritic.__init__() got multiple values for argument 'num_one_step_obs'`

原因：

- `HIMOnPolicyRunner` 初始化 `HIMActorCritic` 时手动传了一次 `num_one_step_obs`
- `policy_cfg` 里又带了一次

处理：

- runner 初始化前先 `pop("num_one_step_obs", None)`，避免重复传参。

### 7.5 `Actor observation dim 256 is not divisible by one-step dim 45`

原因：

- 任务已经启用了 history obs，但配置里还写死 `45`。

处理：

- 让 wrapper 从 observation manager 自动推导 one-step obs dim。
- 若配置值和推导值冲突，以推导值为准，并打印 warning。

### 7.6 Hydra 找不到 HIM 配置

原因：

- 没有在任务注册入口中补 `rsl_rl_him_cfg_entry_point`。

处理：

- 给相关任务的 `__init__.py` 全部补齐 entry point。

### 7.7 HIM 日志不像 PPO

原因：

- 原始 runner 只打印 loss，不消费 `infos["log"]`。

处理：

- 在 `HIMOnPolicyRunner` 中增加 `infos["log"]` 聚合、缓存和格式化打印。


## 8. 建议的任务接入顺序

如果后续继续扩展任务，建议按下面顺序验证：

1. `UnitreeA1 rough`
2. `ATDog Dog3 rough`
3. `stairs / sand / slope`
4. `wheeled`
5. `arm`
6. `humanoid`

原因：

- 标准四足 rough 的观测契约更稳定，最适合先验证算法链路。
- `ATDog` 的任务配置更复杂，直接上业务机器人调试成本更高。


## 9. 后续新增任务时的操作模板

### 9.1 环境侧

确保 observation contract 满足：

- `policy` 只保留单帧本体观测，并通过 history 提供时序输入
- `critic` 保留单帧 privileged obs
- actor 和 critic 的 term 命名尽量保持一致，便于自动推 estimator target slice

### 9.2 配置侧

新增一个任务专属 `rsl_rl_him_cfg.py`，内容至少包括：

- `class_name = "HIMOnPolicyRunner"`
- `policy.class_name = "HIMActorCritic"`
- `algorithm` 沿用 PPO 配置起步

优先调的超参一般是：

- `history_length`
- `entropy_coef`
- `learning_rate`
- `estimator_learning_rate`
- `actor_hidden_dims`
- `critic_hidden_dims`

### 9.3 注册侧

在任务目录的 `__init__.py` 里补：

- `rsl_rl_him_cfg_entry_point`

### 9.4 验证侧

先检查四件事：

1. 训练入口能否正常实例化 `HIMOnPolicyRunner`
2. actor obs 和 critic obs 维度是否正确
3. 日志中是否出现 `Metrics/base_velocity/error_vel_xy` 和 `error_vel_yaw`
4. 日志中是否出现 `error_vel_x` / `error_vel_y` / `error_vel_z` / `error_vel_dir`


## 10. 当前结果

截至 2026-07-25，这次移植已经完成以下能力：

- 本仓本地 HIM 算法栈已接入
- `train.py` / `play.py` / `play_cs.py` 已支持 HIM runner
- `Isaac Lab + Gymnasium` 接口已适配
- Isaac Lab term-major history 已转换为 HIM current-frame-first history
- estimator 切片不再强依赖原始 `HIMLoco` 固定索引
- ATDog 和 Unitree A1 已补 HIM 配置与注册
- 导出链路已改为按真实维度导出
- HIM 终端日志已接近 PPO 风格
- 已补细粒度速度误差指标：
  - `error_vel_x`
  - `error_vel_y`
  - `error_vel_z`
  - `error_vel_dir`
  - `error_vel_xy`
  - `error_vel_yaw`


## 11. 最小启动示例

训练：

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0 \
  --agent=rsl_rl_him_cfg_entry_point
```

如果机器资源紧张，建议加：

```bash
--num_envs=64
```

回放：

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-Velocity-Rough-Unitree-A1-v0 \
  --agent=rsl_rl_him_cfg_entry_point \
  --checkpoint=<checkpoint_path>
```


## 12. 总结

这次移植的核心不是“把 `HIMLoco` 代码拷进来”，而是把它的算法约束重新映射到 `AT_rl_sar` 的任务和观测体系中。真正决定成败的是三点：

- `policy` / `critic` 观测的稳定拆分
- estimator 切片从硬编码改成配置化和自动推导
- 日志、导出、Hydra 注册这些工程链路全部补齐

只要后续继续遵守这三个约束，新增四足任务的 HIM 接入成本会比较低；真正复杂的部分会更多集中在任务观测契约，而不是算法本身。


## 13. 2026-07-26 移植与六帧历史评估

### 13.1 提交评估

- `7d5e5be52aa67a6076a1e516102ff449ec396dd4` 完成了 HIM 算法栈移植，覆盖 actor-critic、estimator、PPO、storage、runner、vec wrapper、导出和任务注册。整体移植方向正确，并且比原始 `HIMLoco` 更适合本仓的一点是 estimator 的 `base_lin_vel` / target slice 会优先从 Robot Lab observation layout 自动推断，而不是继续依赖原始工程的固定 `(45, 48)` / `(3, 48)` 索引。
- `d32ca17d59b8906260176b48065e796086ca7ac5` 把 HIM policy 观测切到 6 帧历史，训练、play、play_cs 都会在 HIM runner 下自动设置 `policy.history_length = 6` 和 `flatten_history_dim = True`。这解决了原始移植后 actor 实际仍可能只吃单帧输入的问题。
- 当前工作区新增部署元数据导出后，HIM 模型导出会额外生成 `policy_metadata.json` 和 AT_robot-lab 历史配置片段，便于实机侧检查 one-step 维度、history 长度、term 顺序和不支持 term。

### 13.2 六帧历史效果

六帧历史对 HIM 是必要项，不只是提高性能的超参。HIM actor 实际输入为：

```text
[current_one_step_obs, estimated_base_lin_vel, latent]
```

其中 `estimated_base_lin_vel` 和 `latent` 都来自 estimator 对完整历史观测的编码。没有稳定的历史输入时，estimator 只能从单帧本体状态猜速度和隐变量，退化明显。

当前实现的历史顺序是：

```text
[t0_all_terms, t-1_all_terms, t-2_all_terms, ..., t-5_all_terms]
```

即 time-major、latest-to-oldest。Isaac Lab 内部历史 buffer 输出 oldest-to-current，`HIMVecEnvWrapper` 会倒序重排后再给 HIM。这个顺序与 AT_robot-lab 的 `ObservationBuffer` 在 `observations_history_priority: "time"`、`observations_history: [0, 1, 2, 3, 4, 5]` 下完全一致。

### 13.3 AT_robot-lab sim2real 适配性

AT_robot-lab 侧已经支持历史观测：

- `observations_history` 中 `0` 表示最新帧，`5` 表示最旧帧。
- `observations_history_priority: "time"` 会按整帧拼接，符合 HIM 导出的 JIT/ONNX 输入。
- `InitRL()` 会用当前观测预填满 history buffer，避免实机启动前几帧喂零历史。

需要注意的部署约束：

- `observations` 顺序必须和训练 policy term 顺序一致。ATDog2 当前训练侧顺序是 `ang_vel, gravity_vec, commands, dof_pos, dof_vel, actions`。
- `num_observations` 必须是一帧维度，HIM ATDog2 为 45；模型实际输入是 `45 * 6 = 270`。
- stairs / sand / bar / slope 等 AT_robot-lab 配置目前仍多为 `observations_history: []`，如果部署 HIM checkpoint，必须改为 `[0, 1, 2, 3, 4, 5]`。

### 13.4 已做优化

- estimator 更新现在跟随 PPO adaptive KL 调度后的 `learning_rate`，与原始 HIMLoco 的训练节奏一致，避免 PPO 学习率已经降/升而 estimator 仍固定在初始 lr。
- AT_robot-lab 配置片段导出文件改为 `at_robot_lab_history.yaml`，不再默认覆盖导出目录下可能已有的正式 `config.yaml`。
- 导出的 AT_robot-lab 片段增加 `policy_input_dim` 注释，便于部署时确认 JIT/ONNX 输入维度是否等于 `num_observations * len(observations_history)`。

### 13.5 后续优化优先级

1. 先用 `--export-only` 对每个 HIM checkpoint 生成 `policy_metadata.json`，确认 `policy_terms` 与 AT_robot-lab YAML 的 `observations` 一致。
2. 把 AT_robot-lab 的 stairs / sand / bar / slope / bridge HIM 策略配置统一开启六帧历史。
3. 训练侧保持 policy 不含 `base_lin_vel`，critic 保留 `base_lin_vel`，否则 estimator 目标会失去 sim2real 意义。
4. 若继续强化粗糙地形能力，优先微调奖励和 domain randomization，不建议把 height scan 加回 policy；加回会提高仿真成绩，但会破坏当前无高度传感器的实机输入契约。


## 14. 2026-07-26 Dog2 抖动问题调参记录

现象：

- 无速度命令时，机身前后上下抖动。
- 有速度命令时，步态推进效果差，前后俯仰/上下振荡明显。

主要判断：

- 零速命令样本比例原来只有 `rel_standing_envs=0.02`，策略很少真正学习“停住”。
- `feet_air_time=50`、`feet_gait=15`、`track_ang_vel_z=40` 偏激进，容易鼓励持续抬腿和强行转向，零速附近也会把策略推向动态步态。
- `upward` 项返回的是姿态偏差平方，正权重会奖励偏差，不适合作为稳身项。
- `flat_orientation_l2`、`ang_vel_xy_l2`、`lin_vel_z_l2` 和 `body_lin_acc_l2` 对前后俯仰/上下振荡约束不足。
- reset 初始 roll/pitch/速度扰动过大，不适合作为当前阶段的稳定步态起点。

已调整的训练起点：

- 零速样本比例提高到 `0.2`。
- 命令范围收窄到 `x=(-0.8, 0.8)`、`y=(-0.25, 0.25)`、`yaw=(-0.6, 0.6)`。
- 降低速度追踪、腾空时间和步态同步奖励，避免为了追踪命令牺牲机身稳定。
- 增强 `stand_still`、`joint_pos_penalty`、`action_rate_l2`、`joint_acc_l2`、`joint_power`。
- 开启 `flat_orientation_l2=-2.0`，加大 `lin_vel_z_l2=-8.0`、`ang_vel_xy_l2=-1.0`，新增 `body_lin_acc_l2=-1e-4`。
- 关闭 `upward`。
- reset 初始姿态/速度扰动收窄，先让策略学会稳定站立和低速步态。
- 补充 `action_smoothness_2_l2` 二阶动作差分惩罚，抑制相邻动作变化方向来回翻转造成的高频抖动。

建议训练流程：

1. 先训练这个保守版本，观察零速站立是否不再周期性点头/弹跳。
2. 如果零速稳定但走得慢，再逐步提高 `track_lin_vel_xy_exp` 到 `30-40`，不要直接回到 `50+`。
3. 如果脚拖地，再把 `feet_air_time` 从 `8` 提到 `12-18`，或把 threshold 从 `0.25` 提到 `0.3`，不要回到 `0.5/50`。
4. 如果台阶能力不足，优先做地形课程或命令课程，而不是把姿态稳定惩罚降掉。

### 14.1 有速度命令时上下抖和非对角步态

新现象：

- 无速度命令时已经基本不抖，说明站立项和零速样本比例有效。
- 有速度命令时仍有机身上下弹跳，且步态没有稳定形成对角 trot。

本轮判断：

- `track_lin_vel_xy_exp=70`、`track_ang_vel_z_exp=50` 对当前阶段偏强，策略容易用弹跳/冲击换速度追踪。
- 仅靠 `feet_gait` 的接触/腾空时间同步约束不够直接，不能强力排除 pacing、bounding 或四脚同相跳。
- 需要把“有命令时的动态稳定”和“当前接触模式必须接近对角步态”分开约束，避免破坏已经稳定的零速站立。

已调整：

- 新增 `diagonal_trot_contact_pattern`，只在命令范数超过阈值时启用，惩罚非 `FL+RR` / `FR+RL` 对角接触模式。
- `feet_gait` 提高到 `12.0`，继续约束对角腿相位同步。
- `track_lin_vel_xy_exp` 降到 `50.0`，`track_ang_vel_z_exp` 降到 `25.0`，减少为追踪命令牺牲机身稳定的倾向。
- `lin_vel_z_l2=-10.0`、`ang_vel_xy_l2=-6.0`、`body_lin_acc_l2=-2e-4`，增强行走时机身上下/俯仰抑振。
- `feet_air_time=5.0`、`feet_air_time_variance=-6.0`，降低鼓励大幅抬腿的强度，同时保持四腿节律一致。

下一轮观察重点：

1. 如果速度明显变慢但步态变成对角步，先保持该配置继续训练，再逐步把 `track_lin_vel_xy_exp` 提到 `55-60`。
2. 如果仍四脚跳或 pacing，把 `diagonal_trot_contact_pattern.weight` 从 `-1.0` 加到 `-1.5`，不要优先提高 `feet_air_time`。
3. 如果对角步态出现但脚拖地，再小幅提高 `feet_air_time` 到 `6-8`，或把 threshold 从 `0.25` 提到 `0.28`。
