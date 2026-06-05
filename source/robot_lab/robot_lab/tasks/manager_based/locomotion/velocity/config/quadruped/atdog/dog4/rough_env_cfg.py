# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# ============================================================================
# ATDog Dog4 粗糙地形速度跟踪环境配置
# ============================================================================
# 本文件定义了四足机器人在台阶地形上进行速度跟踪训练的强化学习环境配置。
# 
# 核心目标：训练机器人能够在上下台阶的复杂地形中稳定行走并跟踪速度指令。
# 
# 关键设计要点：
# 1. 地形配置：使用倒金字塔和正金字塔台阶组合，模拟真实上/下台阶场景
# 2. 观测空间：关闭部分观测（线速度、高度扫描），迫使策略依赖本体感知
# 3. 奖励设计：平衡速度跟踪、能量效率、步态稳定性和关节保护
# 4. 随机化：通过质量、质心、外力扰动提升策略鲁棒性
# ============================================================================

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs import mdp as isaaclab_mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
# # use cloud assets
# from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
# use local assets
from robot_lab.assets.atdog import AT_DOG4_CFG  # isort: skip

# ============================================================================
# 自定义台阶地形配置（倒金字塔上台阶 + 正金字塔下台阶）
# ============================================================================
# 地形参数说明：
# - step_height_range: 台阶高度范围 (9-11cm)，模拟真实楼梯高度
# - step_width: 台阶水平长度 30cm，提供足够的落脚空间
# - platform_width: 平台宽度 3m，在台阶之间提供平坦区域用于调整姿态
# - proportion: 两种台阶地形各占50%，形成交替的上/下台阶序列
# 
# MeshInvertedPyramidStairsTerrainCfg: 倒金字塔台阶（从低到高，上台阶）
# MeshPyramidStairsTerrainCfg: 正金字塔台阶（从高到低，下台阶）
# ============================================================================
DOG4_ROUGH_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),              # 单个地形块尺寸 8m x 8m
    border_width=20.0,            # 边界宽度 20m，防止机器人走出地形
    num_rows=10,                  # 地形网格行数
    num_cols=20,                  # 地形网格列数
    horizontal_scale=0.1,         # 水平分辨率 10cm
    vertical_scale=0.005,         # 垂直分辨率 0.5cm
    slope_threshold=0.75,         # 坡度阈值，超过此值视为不可通行
    use_cache=False,              # 不使用缓存，每次重新生成地形
    sub_terrains={
        "stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0,                    # 占比100%（实际与stairs2各占50%）
            step_height_range=(0.09, 0.11),   # 台阶高度范围 9-11cm（最初0.07-0.13）
            step_width=0.30,                   # 台阶水平长度 30cm
            platform_width=3.0,                # 平台宽度 3m
            border_width=1.0,                  # 边界宽度 1m
            holes=False,                       # 不生成孔洞
        ),
        "stairs2": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0,                    # 占比100%（实际与stairs各占50%）
            step_height_range=(0.09, 0.11),   # 台阶高度范围 9-11cm
            step_width=0.30,                   # 台阶水平长度 30cm
            platform_width=3.0,                # 平台宽度 3m
            border_width=1.0,                  # 边界宽度 1m
            holes=False,                       # 不生成孔洞
        ),
    },
)


@configclass
class ATDogDog4RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """
    ATDog Dog4 粗糙地形速度跟踪环境配置类
    
    继承自 LocomotionVelocityRoughEnvCfg，针对 ATDog Dog4 四足机器人
    在台阶地形上的速度跟踪任务进行定制化配置。
    """
    
    # ==================== 基础标识配置 ====================
    # 机身主刚体名称，用于:
    # 1) 传感器挂载（高度扫描器）
    # 2) 质量/质心/外力随机化时筛选 body
    # 3) 与 base 相关奖励项的 body 指定
    base_link_name = "base"
    
    # URDF 中的 foot link 通过固定关节连接，导入时（merge_fixed_joints=True）
    # 会并入父级 calf body，因此接触相关奖励/惩罚需要匹配末端 calf body。
    # 使用正则表达式匹配所有腿的小腿连杆（.*_calf 匹配 FR_calf, FL_calf, RR_calf, RL_calf）
    foot_link_name = ".*_calf"
    
    # fmt: off
    # ==================== 关节配置 ====================
    # 关节顺序与动作/观测向量顺序保持一致，避免策略输入输出错位
    # 命名规范：{位置}_{关节类型}_joint
    # - 位置: FR(前右), FL(前左), RR(后右), RL(后左)
    # - 关节类型: hip(髋关节), thigh(大腿关节), calf(小腿关节)
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",   # 前右腿
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",   # 前左腿
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",   # 后右腿
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",   # 后左腿
    ]
    # fmt: on

    def __post_init__(self):
        """
        初始化后处理函数
        
        先继承父类默认配置，再按 ATDog Dog4 粗糙地形任务覆写特定参数。
        这是配置类的标准初始化模式，确保所有默认值已设置后再进行定制。
        """
        # 先继承父类默认配置，再按 ATDog Dog4 粗糙地形任务覆写
        super().__post_init__()
        # 保持采样频率不变（sim.dt 与 decimation 沿用父类），仅将单回合时长设为 2s
        # self.episode_length_s = 2.0

        # ==============================Scene 场景与传感器==============================
        # 覆写默认 rough terrain：仅使用固定参数台阶地形
        # 替换父类中的混合地形（平面+斜坡+台阶等）为纯台阶地形
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = DOG4_ROUGH_TERRAIN_CFG
        
        # 指定机器人资产，并放置到每个并行环境的 Robot prim 下
        # {ENV_REGEX_NS} 是 Isaac Lab 的环境命名空间占位符
        self.scene.robot = AT_DOG4_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        
        # 将高度扫描器挂到机身 base 上，保证地形感知参考系一致
        # height_scanner: 大范围扫描（1.6m x 1.0m，分辨率0.1m），用于地形感知
        # height_scanner_base: 小范围扫描（0.1m x 0.1m，分辨率0.05m），用于精确高度测量
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ==============================Observations 观测==============================
        # 观测空间配置决定了策略网络能看到什么信息
        # scale 系数用于归一化不同量纲的观测值，使神经网络训练更稳定
        
        # 各观测分量缩放系数:
        # - scale 越大，对应量在策略输入中的数值幅度越大
        # - 需结合网络归一化习惯，避免某一类信号过强/过弱
        self.observations.policy.base_lin_vel.scale = 2.0      # 机身线速度放大2倍
        self.observations.policy.base_ang_vel.scale = 0.25     # 机身角速度缩小4倍
        self.observations.policy.joint_pos.scale = 1.0         # 关节位置保持原值
        self.observations.policy.joint_vel.scale = 0.05        # 关节速度缩小20倍
        
        # 【重要】关闭策略侧 base_lin_vel 与 height_scan 观测（即不输入给 actor）
        # 说明: 这里常用于做"部分可观测"训练，迫使策略更依赖本体状态（关节角度、角速度等）
        # 这样做的好处：
        # 1. 提高策略对传感器故障的鲁棒性（真实环境中可能没有精确的速度计或高度计）
        # 2. 促使策略学习基于本体感知的运动控制，更接近生物的运动方式
        # 3. 减少观测维度，降低网络复杂度
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        
        # 为策略侧启用 10 帧历史观测。
        # Isaac Lab 会按 term 展开历史后再拼接，和 rl_sar 的 observations_history_priority="term" 对齐。
        # 历史观测的作用：
        # 1. 提供时间序列信息，帮助策略理解运动趋势
        # 2. 弥补部分可观测性的不足（如没有线速度观测时，可通过位置变化推断速度）
        # 3. 增强策略对动态变化的响应能力
        self.observations.policy.history_length = 10
        self.observations.policy.flatten_history_dim = True  # 将历史维度展平，便于网络处理
        
        # 为了与历史 checkpoint 的网络输入维度保持一致，同时关闭 critic 侧高度扫描观测
        # 否则 resume 时会出现 critic 第一层权重 shape mismatch（例如 48 vs 235）
        # Critic 网络通常拥有完整观测（privileged information），但这里为了兼容性也关闭了高度扫描
        self.observations.critic.height_scan = None
        
        # 明确 joint_pos/joint_vel 仅采集 joint_names 中定义的关节
        # 这确保观测向量的顺序和长度与动作向量严格对应
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        
        # 添加姿态四元数观测，因为真实狗IMU能反馈角度（四元数形式）
        # 注：当前已注释，如需启用可取消注释
        # 优势：提供更精确的姿态信息，有助于在崎岖地形上保持平衡
        # 劣势：增加观测维度，可能需要更多训练数据
        # self.observations.policy.base_orientation = ObsTerm(
        #     func=isaaclab_mdp.root_quat_w,
        #     noise=Unoise(n_min=-0.01, n_max=0.01),
        #     clip=(-1.0, 1.0),
        #     scale=1.0,
        # )
        
        # 添加关节力矩观测，因为电机能反馈力矩
        # 注：当前已注释，如需启用可取消注释
        # 优势：提供负载信息，帮助策略判断地形阻力
        # 劣势：力矩噪声较大，可能需要更强的滤波
        # self.observations.policy.joint_effort = ObsTerm(
        #     func=isaaclab_mdp.joint_effort,
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names)},
        #     noise=Unoise(n_min=-0.01, n_max=0.01),
        #     clip=(-100.0, 100.0),
        #     scale=0.01,
        # )

        # ==============================Actions 动作==============================
        # 动作空间配置决定了策略能控制什么以及如何控制
        
        # 动作缩放:
        # - 髋关节幅度更小(0.125)，减少横摆过猛导致的不稳定
        # - 其余关节幅度0.25，保留足够摆动能力
        # 这里的 key 是正则，按关节名匹配后应用对应 scale
        # 为什么髋关节要更小？
        # 1. 髋关节控制左右摆动，过大的动作会导致机身剧烈摇晃
        # 2. 大腿和小腿关节主要控制前后运动，需要更大的活动范围来跨越台阶
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        
        # 动作裁剪区间（非常宽），主要作为安全兜底防止异常值爆炸
        # 正常情况下动作不会接近这个范围，但设置一个极大的限制可以防止数值错误
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        
        # 限定动作控制关节集合，顺序与 joint_names 对齐
        # 确保动作向量的每个元素对应正确的关节
        self.actions.joint_pos.joint_names = self.joint_names

        # ==============================Events 随机化事件==============================
        # 域随机化（Domain Randomization）是提升策略泛化能力的关键技术
        # 通过在训练中引入各种随机扰动，使策略能够适应真实世界的不确定性
        
        # reset 时随机化机身位姿与速度:
        # - pose_range: 初始位置/姿态扰动范围
        # - velocity_range: 初始线速度/角速度扰动范围
        # 目的: 提升鲁棒性，减少对单一起始状态过拟合
        # 注意：当前 yaw/pitch/roll 均为0，表示只随机化位置，不随机化初始姿态
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),    # X方向位置随机 ±0.5m
                "y": (-0.5, 0.5),    # Y方向位置随机 ±0.5m
                "z": (0.0, 0.2),     # Z方向位置随机 0~0.2m（只能向上偏移，避免陷入地面）
                "roll": (-0.0, 0.0), # Roll角不随机
                "pitch": (-0.0, 0.0),# Pitch角不随机
                "yaw": (-0.0, 0.0),  # Yaw角不随机
            },
            "velocity_range": {
                "x": (-0.0, 0.0),    # 初始线速度不随机
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0), # 初始角速度不随机
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        }
        
        # 仅随机化 base 的质量
        # 通过改变机身质量，让策略适应不同的负载情况
        # 父类配置中 mass_distribution_params=(-1.0, 3.0)，operation="add"
        # 表示在原始质量基础上增加 -1.0~3.0 kg
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        
        # 随机化除 base 外其他刚体质量（负向前瞻正则: 不匹配 base）
        # 腿部质量变化会影响惯性和动力学特性
        # 父类配置中 mass_distribution_params=(0.7, 1.3)，operation="scale"
        # 表示在原始质量基础上乘以 0.7~1.3 的系数
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        
        # 仅随机化 base 质心位置
        # 质心偏移会显著影响平衡控制难度
        # 父类配置中 com_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}
        # 表示质心在各方向上最多偏移 5cm
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        
        # 外力/外力矩扰动施加到 base，模拟推搡/干扰
        # 父类配置中 force_range=(-10.0, 10.0)，torque_range=(-10.0, 10.0)
        # 这会让策略学会抵抗外部干扰，提高稳定性
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ==============================Rewards 奖励函数================================================================================================================================================
        # 奖励函数是强化学习的核心，它告诉智能体什么是"好"的行为
        # 
        # 奖励类型说明：
        # - weight > 0: 奖励（Reward），鼓励智能体执行该行为
        # - weight < 0: 惩罚（Penalty），阻止智能体执行该行为
        # - weight = 0: 禁用该项（通过 disable_zero_weight_rewards 自动清理）
        #
        # 奖励设计原则：
        # 1. 稀疏奖励 vs 密集奖励：这里采用密集奖励（每步都计算），加速收敛
        # 2. 奖励 shaping：通过多个子奖励组合，引导策略学习期望行为
        # 3. 量纲平衡：通过调整 weight 和 scale，使各项奖励在同一数量级
        
        # ------------------------------ General 通用奖励 ------------------------------
        # 终止状态惩罚：理论上应该被 terminations 处理，这里设为0避免重复计算
        self.rewards.is_terminated.weight = 0

        # ------------------------------ Root penalties 机身惩罚 ------------------------------
        # 这些惩罚项约束机身整体的运动状态，确保稳定性和安全性
        
        # 【惩罚】Z轴线速度的L2范数
        # 作用：抑制机身在垂直方向的跳动，鼓励平稳的水平运动
        # 原理：四足机器人主要在XY平面运动，Z轴速度过大会浪费能量且不稳定
        # weight = -2.0：中等强度惩罚
        self.rewards.lin_vel_z_l2.weight = -2.0
        
        # 【惩罚】XY平面角速度的L2范数
        # 作用：抑制机身绕X/Y轴的旋转（roll/pitch），保持机身水平
        # 原理：过度的倾斜会导致摔倒或能量损失
        # weight = -0.05：较弱惩罚，允许一定的姿态调整
        self.rewards.ang_vel_xy_l2.weight = -0.05
        
        # 【禁用】机身姿态惩罚（L2范数）
        # 原本用于惩罚偏离水平姿态，但这里设为0
        # 原因：可能与其他奖励（如upward）功能重叠，或通过其他方式约束
        self.rewards.flat_orientation_l2.weight = 0
        
        # 【禁用】机身高度惩罚（L2范数）
        # 原本用于惩罚偏离目标高度，但这里设为0
        # 原因：可能通过 feet_height_body 等其他奖励间接控制高度
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0.33  # 目标高度33cm
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        
        # 【禁用】机身线性加速度惩罚（L2范数）
        # 原本用于平滑运动，减少剧烈加减速
        # 原因：可能通过 action_rate_l2 等动作平滑项间接控制
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------ Joint penalties 关节惩罚 ------------------------------
        # 这些惩罚项约束关节运动，保护电机并提高能量效率
        
        # 【惩罚】关节力矩的L2范数
        # 作用：鼓励节能，减少电机发热和磨损
        # 原理：力矩与电流成正比，减小力矩可降低能耗
        # weight = -2.5e-5：微弱惩罚，因为力矩主要由任务需求决定
        self.rewards.joint_torques_l2.weight = -1.5e-5
        
        # 【禁用】关节速度L2范数惩罚
        # 原本用于限制关节运动速度
        # 原因：可能通过 joint_vel_limits 或 action_rate_l2 间接控制
        self.rewards.joint_vel_l2.weight = 0
        
        # 【惩罚】关节加速度L2范数
        # 作用：平滑关节运动，减少机械冲击和振动
        # 原理：加速度与力成正比，减小加速度可降低机械应力
        # weight = -2.5e-7：极弱惩罚，仅作为软约束
        self.rewards.joint_acc_l2.weight = -2.5e-7
        
        # 【禁用】髋关节偏离惩罚
        # 原本用于惩罚髋关节偏离默认位置
        # 原因：已通过 joint_pos_penalty 统一处理所有关节
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        
        # 【惩罚】关节位置超出限制
        # 作用：防止关节运动到机械限位附近，保护硬件
        # 原理：当关节角度接近 limits 时施加惩罚
        # weight = -5.0：较强惩罚，因为超限可能导致硬件损坏
        self.rewards.joint_pos_limits.weight = -5.0
        
        # 【禁用】关节速度超出限制
        # 原本用于防止关节超速
        # 原因：可能通过其他方式（如电机模型）隐式约束
        self.rewards.joint_vel_limits.weight = 0
        
        # 【惩罚】关节功率（力矩×速度）
        # 作用：直接优化能量消耗
        # 原理：功率 = 力矩 × 速度，代表瞬时能耗
        # weight = -2e-5：微弱惩罚，与 joint_torques_l2 配合使用
        self.rewards.joint_power.weight = -2e-5
        
        # 【惩罚】静止时的关节运动
        # 作用：当速度指令很小时，鼓励关节保持默认位置
        # 原理：避免在不需要运动时产生不必要的抖动
        # weight = -2.0：中等强度惩罚
        # 注意：仅在命令速度 < 0.1 时生效（见 stand_still 函数实现）
        self.rewards.stand_still.weight = -3.0
        
        # 【惩罚】关节位置偏离默认值
        # 作用：鼓励关节保持在默认姿态附近，除非任务需要
        # 原理：默认姿态通常是能量最优或最稳定的构型
        # weight = -1.0：中等强度惩罚
        # 注意：当速度较大时惩罚减弱，允许必要的关节运动（见 joint_pos_penalty 实现）
        self.rewards.joint_pos_penalty.weight = -1.0
        
        # 【惩罚】左右对称关节的运动不对称
        # 作用：鼓励对称步态，提高运动效率和美观度
        # 原理：对称的步态通常更稳定且能耗更低
        # weight = -0.05：较弱惩罚，允许一定的不对称以适应地形
        # mirror_joints 定义了对称关节对：
        # - FR（前右）↔ RL（后左）：对角对称
        # - FL（前左）↔ RR（后右）：对角对称
        self.rewards.joint_mirror.weight = 0.0
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],  # 前右 ↔ 后左
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],  # 前左 ↔ 后右
        ]

        # ------------------------------ Action penalties 动作惩罚 ------------------------------
        # 这些惩罚项约束动作的变化，提高控制的平滑性
        
        # 【惩罚】动作变化率的L2范数
        # 作用：鼓励平滑的动作输出，减少高频抖动
        # 原理：action_rate = current_action - previous_action
        # weight = -0.05：较弱惩罚，允许必要的快速响应
        # 这对真实机器人很重要，因为高频动作会导致电机过热和机械磨损
        self.rewards.action_rate_l2.weight = -0.08

        # ------------------------------ Contact sensor 接触传感器惩罚 ------------------------------
        # 这些惩罚项基于接触传感器数据，约束机器人与地面的交互
        
        # 【惩罚】非足部身体部位与地面接触
        # 作用：防止机身、腿部等非足部触地，避免摔倒
        # 原理：检测除足部外的所有身体部位的接触力
        # weight = -1.0：中等强度惩罚
        # sensor_cfg.body_names 使用正则排除足部：f"^(?!.*{self.foot_link_name}).*"
        # 即匹配所有不包含 "_calf" 的身体部位
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        
        # 【惩罚】足部接触力过大
        # 作用：防止足部以过大的力撞击地面，减少冲击
        # 原理：当接触力超过阈值（100N）时施加惩罚
        # weight = -1.5e-4：微弱惩罚，因为接触力主要由体重决定
        self.rewards.contact_forces.weight = -1.0e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # ------------------------------ Velocity-tracking rewards 速度跟踪奖励 ------------------------------
        # 这些奖励项鼓励机器人跟踪给定的速度指令，是任务的核心目标
        
        # 【奖励】线速度跟踪（XY平面）
        # 作用：鼓励机器人按照指令速度前进
        # 原理：使用指数核函数 exp(-error²/std²)，误差越小奖励越高
        # weight = 3.0：较强奖励，这是主要任务目标
        # std = sqrt(0.25) = 0.5：标准差，控制奖励的宽容度
        # 注意：当机身严重倾斜时（projected_gravity_b[:, 2] < -0.7），奖励会被抑制
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        
        # 【奖励】角速度跟踪（Z轴旋转）
        # 作用：鼓励机器人按照指令角速度转向
        # 原理：同样使用指数核函数
        # weight = 1.5：中等强度奖励，比线速度略低
        # 转向通常不如前进重要，所以权重较低
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # ------------------------------ Others 其他奖励 ------------------------------
        # 这些奖励项鼓励特定的步态特征和运动模式
        
        # 【奖励】足部空中时间
        # 作用：鼓励抬脚迈步，而不是拖地行走
        # 原理：当足部离地时间超过阈值（0.5s）时给予奖励
        # weight = 0.1：较弱奖励，作为辅助目标
        # 这有助于形成清晰的步态，避免蹭地导致的能量损失和磨损
        # 注意：仅在命令速度 > 0.1 时生效
        self.rewards.feet_air_time.weight = 2.0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        
        # 【惩罚】足部空中时间的方差
        # 作用：鼓励四条腿的空中时间保持一致，形成规律步态
        # 原理：计算四条腿 last_air_time 和 last_contact_time 的方差
        # weight = -1.0：中等强度惩罚
        # 均匀的步态通常更稳定且能耗更低
        self.rewards.feet_air_time_variance.weight = -1.5
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        
        # 【禁用】足部接触奖励
        # 原本用于鼓励特定数量的足部接触
        # 原因：可能与 feet_contact_without_cmd 功能重叠
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        
        # 【奖励】静止时的足部接触
        # 作用：当速度指令很小时，鼓励足部保持接触（站立稳定）
        # 原理：计算接触足部的数量，越多越好
        # weight = 0.1：较弱奖励
        # 注意：仅在命令速度 < 0.1 时生效（见 feet_contact_without_cmd 函数实现）
        self.rewards.feet_contact_without_cmd.weight = 0.05
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        
        # 【禁用】足部绊倒惩罚
        # 原本用于惩罚足部侧面受力（撞到障碍物）
        # 原因：可能在台阶地形中误判（正常上台阶时侧面也会受力）
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        
        # 【惩罚】足部滑动
        # 作用：防止足部在地面上滑动（应抬起移动）
        # 原理：当足部接触地面且有水平速度时施加惩罚
        # weight = -0.1：较弱惩罚
        # 滑动会浪费能量且磨损足部
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        
        # 【禁用】足部高度奖励
        # 原本用于鼓励足部抬到特定高度
        # 原因：可能通过其他方式（如轨迹跟踪）间接控制
        self.rewards.feet_height.weight = 2.0
        self.rewards.feet_height.params["target_height"] = 0.10
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        
        # 【惩罚】足部相对于机身的高度
        # 作用：防止足部抬得过高（相对于机身）
        # 原理：惩罚足部位置偏离目标高度（-0.2m，即机身下方20cm）
        # weight = -5.0：较强惩罚
        # 这限制了步幅，避免过度抬腿导致的不稳定
        # 注意：target_height 为负值，表示在机身坐标系中向下
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        
        # 【奖励】步态同步性
        # 作用：鼓励对角腿同步运动（trotting 步态）
        # 原理： penalize 非同步足对的接触时间差异
        # weight = 0.5：中等强度奖励
        # synced_feet_pair_names 定义了应该同步的足对：
        # - (FL_calf, RR_calf)：前左和后右应该同步
        # - (FR_calf, RL_calf)：前右和后左应该同步
        # 这是典型的 trotting 步态模式（对角腿同时着地）
        self.rewards.feet_gait.weight = 1.0
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_calf", "RR_calf"), ("FR_calf", "RL_calf"))
        
        # 【奖励】机身朝上
        # 作用：鼓励机身保持直立姿态
        # 原理：基于 projected_gravity 向量计算，越接近竖直向上奖励越高
        # weight = 1.0：中等强度奖励
        # 这是基本的稳定性要求，防止机器人翻倒
        self.rewards.upward.weight = 1.0

        # 将权重为0的奖励项禁用，减少无效计算与配置噪声
        # 这是一个优化措施，自动清理未使用的奖励项
        if self.__class__.__name__ == "ATDogDog4RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations 终止条件------------------------------
        # 终止条件定义了 episode 何时结束（除了超时和走出边界）
        
        # base 接触地面时终止，避免策略学会趴着蹭地获得速度奖励。
        # 这是一个重要的安全措施：
        # 1. 防止策略发现"漏洞"：通过让机身贴地滑动来获得速度奖励
        # 2. 确保学习到的是真正的行走行为
        # 3. 保护硬件，避免机身摩擦地面
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]

        # ------------------------------Curriculums 课程学习------------------------------
        # 课程学习通过逐渐增加任务难度来提升训练效果
        
        # self.curriculum.command_levels.params["range_multiplier"] = (0.2, 1.0)
        # 关闭命令课程，直接使用固定命令范围
        # 原因：可能希望从一开始就训练全速度范围，或手动控制难度
        self.curriculum.command_levels = None

        # ------------------------------Commands 命令范围------------------------------
        # 速度指令的范围定义
        # 如需限制速度指令范围，可取消下方注释并按需调整
        # 当前使用父类默认值：lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
