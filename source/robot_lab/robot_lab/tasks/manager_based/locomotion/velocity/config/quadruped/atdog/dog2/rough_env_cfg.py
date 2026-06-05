# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0
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
from robot_lab.assets.atdog import AT_DOG2_CFG  # isort: skip

# 自定义台阶地形（倒金字塔上台阶）:
# - 台阶高 10cm
# - 台阶水平长度 30cm
# 说明:
# 使用 MeshInvertedPyramidStairsTerrainCfg（与 Isaac Lab 默认 rough 配置一致）
# 生成倒金字塔台阶，机器人可从低处向高处持续上台阶。
DOG2_ROUGH_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "stairs": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.09, 0.11),#(最初0.07-0.13)
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs2": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.09, 0.11),
            step_width=0.30,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


@configclass
class ATDogDog2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    # 机身主刚体名称，用于:
    # 1) 传感器挂载（高度扫描器）
    # 2) 质量/质心/外力随机化时筛选 body
    # 3) 与 base 相关奖励项的 body 指定
    base_link_name = "base"
    # URDF 中的 foot link 通过固定关节连接，导入时（merge_fixed_joints=True）
    # 会并入父级 calf body，因此接触相关奖励/惩罚需要匹配末端 calf body。
    foot_link_name = ".*_calf"
    # fmt: off
    # 关节顺序与动作/观测向量顺序保持一致，避免策略输入输出错位
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    # fmt: on

    def __post_init__(self):
        # 先继承父类默认配置，再按 ATDog Dog2 粗糙地形任务覆写
        super().__post_init__()
        # 保持采样频率不变（sim.dt 与 decimation 沿用父类），仅将单回合时长设为 2s
        #self.episode_length_s = 2.0

        # ------------------------------Scene 场景与传感器------------------------------
        # 覆写默认 rough terrain：仅使用固定参数台阶地形
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = DOG2_ROUGH_TERRAIN_CFG
        # 指定机器人资产，并放置到每个并行环境的 Robot prim 下
        self.scene.robot = AT_DOG2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 将高度扫描器挂到机身 base 上，保证地形感知参考系一致
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations 观测------------------------------
        # 各观测分量缩放系数:
        # - scale 越大，对应量在策略输入中的数值幅度越大
        # - 需结合网络归一化习惯，避免某一类信号过强/过弱
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        # 关闭策略侧 base_lin_vel 与 height_scan 观测（即不输入给 actor）
        # 说明: 这里常用于做“部分可观测”训练，迫使策略更依赖本体状态
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        # 为策略侧启用 5 帧历史观测。
        # Isaac Lab 会按 term 展开历史后再拼接，和 rl_sar 的 observations_history_priority="term" 对齐。
        self.observations.policy.history_length = 5
        self.observations.policy.flatten_history_dim = True
        # 为了与历史 checkpoint 的网络输入维度保持一致，同时关闭 critic 侧高度扫描观测
        # 否则 resume 时会出现 critic 第一层权重 shape mismatch（例如 48 vs 235）
        self.observations.critic.height_scan = None
        # 明确 joint_pos/joint_vel 仅采集 joint_names 中定义的关节
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        # 添加姿态四元数观测，因为真实狗IMU能反馈角度（四元数形式）
        # self.observations.policy.base_orientation = ObsTerm(
        #     func=isaaclab_mdp.root_quat_w,
        #     noise=Unoise(n_min=-0.01, n_max=0.01),
        #     clip=(-1.0, 1.0),
        #     scale=1.0,
        # )
        # # 添加关节力矩观测，因为电机能反馈力矩
        # self.observations.policy.joint_effort = ObsTerm(
        #     func=isaaclab_mdp.joint_effort,
        #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names)},
        #     noise=Unoise(n_min=-0.01, n_max=0.01),
        #     clip=(-100.0, 100.0),
        #     scale=0.01,
        # )

        # ------------------------------Actions 动作------------------------------
        # 动作缩放:
        # - 髋关节幅度更小(0.125)，减少横摆过猛导致的不稳定
        # - 其余关节幅度0.25，保留足够摆动能力
        # 这里的 key 是正则，按关节名匹配后应用对应 scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        # 动作裁剪区间（非常宽），主要作为安全兜底防止异常值爆炸
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        # 限定动作控制关节集合，顺序与 joint_names 对齐
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events 随机化事件------------------------------
        # reset 时随机化机身位姿与速度:
        # - pose_range: 初始位置/姿态扰动范围
        # - velocity_range: 初始线速度/角速度扰动范围
        # 目的: 提升鲁棒性，减少对单一起始状态过拟合
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        # 仅随机化 base 的质量
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        # 随机化除 base 外其他刚体质量（负向前瞻正则: 不匹配 base）
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        # 仅随机化 base 质心位置
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        # 外力/外力矩扰动施加到 base，模拟推搡/干扰
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0.33
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = -2e-5
        self.rewards.stand_still.weight = -2.0
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.1

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # Others
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight = -1.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = -5.0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0.5
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        self.rewards.upward.weight = 1.0

        # 将权重为0的奖励项禁用，减少无效计算与配置噪声
        if self.__class__.__name__ == "ATDogDog2RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations 终止条件------------------------------
        # base 接触地面时终止，避免策略学会趴着蹭地获得速度奖励。
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name]

        # ------------------------------Curriculums 课程学习------------------------------
        # self.curriculum.command_levels.params["range_multiplier"] = (0.2, 1.0)
        # 关闭命令课程，直接使用固定命令范围
        self.curriculum.command_levels = None

        # ------------------------------Commands 命令范围------------------------------
        # 如需限制速度指令范围，可取消下方注释并按需调整
        # self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
