# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    ActionsCfg,
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
)

##
# Pre-defined configs
##
from robot_lab.assets.atdog import AT_DOG_ARM_CFG  # isort: skip


@configclass
class ATDogArmActionsCfg(ActionsCfg):
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[""], scale=0.25, use_default_offset=True, clip=None, preserve_order=True
    )

    joint_vel = mdp.JointVelocityActionCfg(
        asset_name="robot", joint_names=[""], scale=5.0, use_default_offset=True, clip=None, preserve_order=True
    )


@configclass
class ATDogArmRewardsCfg(RewardsCfg):
    """Reward terms for the MDP."""

    joint_vel_wheel_l2 = RewTerm(
        func=mdp.joint_vel_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_acc_wheel_l2 = RewTerm(
        func=mdp.joint_acc_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )

    joint_torques_wheel_l2 = RewTerm(
        func=mdp.joint_torques_l2, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="")}
    )


@configclass
class ATDogArmRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    actions: ATDogArmActionsCfg = ATDogArmActionsCfg()
    rewards: ATDogArmRewardsCfg = ATDogArmRewardsCfg()

    # 机身主刚体名称，用于:
    # 1) 传感器挂载（高度扫描器）
    # 2) 质量/质心/外力随机化时筛选 body
    # 3) 与 base 相关奖励项的 body 指定
    base_link_name = "base_link"
    # 轮足末端 body，用于接触相关奖励/惩罚筛选
    foot_link_name = ".*_foot"

    # fmt: off
    # 腿部关节顺序与动作/观测向量顺序保持一致，避免策略输入输出错位
    leg_joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    # 轮关节集合（速度控制）
    wheel_joint_names = [
        "FR_foot_joint", "FL_foot_joint", "RR_foot_joint", "RL_foot_joint",
    ]
    # 全关节集合（观测使用）
    joint_names = leg_joint_names + wheel_joint_names
    # fmt: on

    def __post_init__(self):
        # 先继承父类默认配置，再按 ATDog 轮式粗糙地形任务覆写
        super().__post_init__()

        # ------------------------------Scene 场景与传感器------------------------------
        # 指定机器人资产，并放置到每个并行环境的 Robot prim 下
        self.scene.robot = AT_DOG_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 将高度扫描器挂到机身 base 上，保证地形感知参考系一致
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        # 对应 dog2 配置思路:
        # - 仍保留 joint_pos/joint_vel 主观测
        # - 轮式任务中将 joint_pos 替换为不含轮关节的位置观测，避免轮转角累积带来的输入漂移
        self.observations.policy.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.policy.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        self.observations.critic.joint_pos.func = mdp.joint_pos_rel_without_wheel
        self.observations.critic.joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.wheel_joint_names
        )
        # 各观测分量缩放系数:
        # - scale 越大，对应量在策略输入中的数值幅度越大
        # - 需结合网络归一化习惯，避免某一类信号过强/过弱
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        # 关闭策略侧 base_lin_vel 与 height_scan 观测（即不输入给 actor）
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        # 对应 dog2: 显式关闭 critic 侧高度扫描观测，便于保持输入维度一致
        self.observations.critic.height_scan = None
        # 明确 joint_pos/joint_vel 仅采集 joint_names 中定义的关节
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # 动作缩放:
        # - 髋关节幅度更小(0.125)，减少横摆过猛导致的不稳定
        # - 其余腿关节幅度 0.25，保留足够摆动能力
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        # 轮关节速度动作缩放
        self.actions.joint_vel.scale = 5.0
        # 动作裁剪区间（非常宽），主要作为安全兜底防止异常值爆炸
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_vel.clip = {".*": (-100.0, 100.0)}
        # 腿关节走位置控制，轮关节走速度控制
        self.actions.joint_pos.joint_names = self.leg_joint_names
        self.actions.joint_vel.joint_names = self.wheel_joint_names

        # ------------------------------Events------------------------------
        # reset 时随机化机身位姿与速度:
        # - pose_range: 初始位置/姿态扰动范围
        # - velocity_range: 初始线速度/角速度扰动范围
        # 目的: 提升鲁棒性，减少对单一起始状态过拟合
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
                "z": (0.0, 0.02),
                "roll": (-0.8, 0.8),
                "pitch": (-0.8, 0.8),
                "yaw": (-0.1, 0.1),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
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
        # 终止项：通常用于 episode 提前结束时施加一次性惩罚；这里权重为 0，表示当前不启用
        self.rewards.is_terminated.weight = 0

        # Root penalties
        # 惩罚：抑制机身 z 方向线速度，减少上下跳动，鼓励更平稳贴地运动
        self.rewards.lin_vel_z_l2.weight = -2.0
        # 惩罚：抑制机身 roll/pitch 角速度，减少侧翻和前后晃动
        self.rewards.ang_vel_xy_l2.weight = -0.05
        # 惩罚：约束机身姿态接近水平；这里权重为 0，表示当前不直接约束横滚/俯仰姿态
        self.rewards.flat_orientation_l2.weight = 0
        # 惩罚：约束机身高度接近目标高度；这里权重为 0，表示当前不启用高度误差项
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0.40
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        # 惩罚：抑制机身线加速度，减少机身抖动和冲击；这里权重为 0，表示当前不启用
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        # 惩罚：约束腿部输出力矩，降低能耗并减少过猛驱动
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        # 惩罚：约束轮关节输出力矩；这里权重为 0，表示当前不惩罚轮力矩
        self.rewards.joint_torques_wheel_l2.weight = -2.5e-2
        self.rewards.joint_torques_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # 惩罚：约束腿部关节速度，避免动作过快；这里权重为 0，表示当前不启用
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        # 惩罚：约束轮关节速度；这里权重为 0，表示当前允许轮子自由转速，不额外惩罚
        self.rewards.joint_vel_wheel_l2.weight = -5.0e-2
        self.rewards.joint_vel_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # 惩罚：约束腿部关节加速度，抑制控制突变，提升动作平滑性
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = self.leg_joint_names
        # 惩罚：约束轮关节加速度，减少轮速突变；权重较小，主要做轻微平滑正则
        self.rewards.joint_acc_wheel_l2.weight = -2.5e-9
        self.rewards.joint_acc_wheel_l2.params["asset_cfg"].joint_names = self.wheel_joint_names
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        # 惩罚：腿部关节接近位置极限时扣分，避免打到关节边界
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_pos_limits.params["asset_cfg"].joint_names = self.leg_joint_names
        # 惩罚：轮关节接近速度极限时扣分；这里权重为 0，表示当前不启用
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_vel_limits.params["asset_cfg"].joint_names = self.wheel_joint_names
        # 惩罚：按功率消耗扣分，鼓励更省力的腿部运动
        self.rewards.joint_power.weight = -2e-5
        self.rewards.joint_power.params["asset_cfg"].joint_names = self.leg_joint_names
        # 惩罚：在应保持静止或低速时仍大幅摆腿会扣分，抑制原地乱动
        self.rewards.stand_still.weight = -1.5
        self.rewards.stand_still.params["asset_cfg"].joint_names = self.leg_joint_names
        # 惩罚：腿部关节位置偏离参考姿态时扣分，鼓励保持较自然、稳定的默认构型
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = self.leg_joint_names
        # 惩罚：轮子在不合适接触状态下的转动行为；这里权重为 0，表示当前不启用
        self.rewards.wheel_vel_penalty.weight = 0
        self.rewards.wheel_vel_penalty.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.wheel_vel_penalty.params["asset_cfg"].joint_names = self.wheel_joint_names
        # 惩罚：约束对角腿动作镜像一致性，减少左右/前后不协调动作
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Action penalties
        # 惩罚：约束相邻时刻动作变化率，抑制控制抖动，提升策略输出平滑性
        self.rewards.action_rate_l2.weight = -0.3

        # Contact sensor
        # 惩罚：非足端/轮端 body 出现接触时扣分，减少机身、髋部等不期望碰撞
        self.rewards.undesired_contacts.weight = -2.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        # 惩罚：足端接触力过大时扣分，鼓励更柔和的落地与支撑
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        # 奖励：鼓励机身 x/y 平面线速度跟踪指令，是主要前进/平移任务奖励之一
        self.rewards.track_lin_vel_xy_exp.weight = 20.0
        # 奖励：鼓励机身 z 轴角速度跟踪指令，是主要转向任务奖励之一
        self.rewards.track_ang_vel_z_exp.weight = 16.0

        # Others
        # 奖励：鼓励足端具有合适腾空时间，常用于步态节律学习；这里权重为 0，表示当前不启用
        self.rewards.feet_air_time.weight = 20.0
        self.rewards.feet_air_time.params["threshold"] = 0.3
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 奖励：鼓励足端接触事件本身；这里权重为 0，表示当前不启用
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 奖励：在无速度指令或低指令时鼓励足端接触地面，帮助机器人稳定站立
        self.rewards.feet_contact_without_cmd.weight = 0.05
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 惩罚：足端被障碍绊到时扣分；这里权重为 0，表示当前不启用
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 惩罚：足端与地面接触时发生滑移会扣分；这里权重为 0，表示当前不启用
        self.rewards.feet_slide.weight = -0.25
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        # 奖励/惩罚：约束足端绝对高度接近目标高度；这里权重为 0，表示当前不启用抬脚高度约束
        self.rewards.feet_height.weight = -0.07
        self.rewards.feet_height.params["target_height"] = 0.20
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        # 奖励/惩罚：约束足端相对机身的高度关系；这里权重为 0，表示当前不启用
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        # 奖励：鼓励形成指定对角同步步态；这里权重为 0，表示当前不显式约束步态型态
        self.rewards.feet_gait.weight = 5
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        # 奖励：鼓励机身朝上，维持整体竖直稳定姿态
        self.rewards.upward.weight = 1.0


        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "ATDogArmRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels = None

        # ------------------------------Commands------------------------------
        # self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5)
        # self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
