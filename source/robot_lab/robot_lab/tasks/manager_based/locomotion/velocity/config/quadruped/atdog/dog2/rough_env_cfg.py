# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
# # use cloud assets
# from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
# use local assets
from robot_lab.assets.atdog import AT_DOG2_CFG  # isort: skip


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

        # ------------------------------Scene 场景与传感器------------------------------
        # 指定机器人资产，并放置到每个并行环境的 Robot prim 下
        self.scene.robot = AT_DOG2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # 盲走设置: 关闭地形高度扫描器，避免任何显式地形感知输入
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None

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
        # 同时关闭 critic 侧 height_scan，避免训练时通过特权观测引入地形感知
        self.observations.critic.height_scan = None
        # 明确 joint_pos/joint_vel 仅采集 joint_names 中定义的关节
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

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
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.03, 0.03),
            "y": (-0.03, 0.03),
            "z": (0.0, 0.02),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
            "yaw": (-0.1, 0.1),
        }
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
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

        # ------------------------------Rewards 奖励函数------------------------------
        # 约定:
        # - 正权重: 鼓励该行为（reward）
        # - 负权重: 惩罚该行为（penalty）
        # - 权重为0: 本项不参与训练（后面可统一 disable）

        # General 通用项
        # 终止惩罚在该配置中关闭（=0）
        self.rewards.is_terminated.weight = 0

        # Root penalties 机身状态惩罚
        # 惩罚竖直方向线速度，减少机身上下蹿动
        self.rewards.lin_vel_z_l2.weight = -3.0
        # 惩罚机身滚转/俯仰角速度，抑制侧翻和点头
        self.rewards.ang_vel_xy_l2.weight = -0.05
        # 机身水平姿态惩罚关闭
        self.rewards.flat_orientation_l2.weight = 0
        # 机身高度惩罚关闭（但保留目标高度参数便于后续开启）
        self.rewards.base_height_l2.weight = -0.0  #（改了0）
        # 机身目标高度（单位: 米）
        self.rewards.base_height_l2.params["target_height"] = 0.27 #（改了0.33）
        # 指定高度项作用 body 为 base
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        # 机身线加速度惩罚关闭
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties 关节相关惩罚
        # 扭矩 L2 惩罚，抑制高能耗/激烈驱动
        self.rewards.joint_torques_l2.weight = -2.5e-5
        # 关节速度惩罚关闭
        self.rewards.joint_vel_l2.weight = 0.0
        # 关节加速度惩罚，减少动作抖动
        self.rewards.joint_acc_l2.weight = -5.0e-6
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        # 关节接近位置极限时惩罚
        self.rewards.joint_pos_limits.weight = -5.0
        # 关节速度上限惩罚关闭
        self.rewards.joint_vel_limits.weight = 0
        # 功率惩罚（近似能耗约束）
        self.rewards.joint_power.weight = -2e-5
        # 有速度命令时站立不动会受惩罚
        self.rewards.stand_still.weight = -10.0
        # 关节姿态偏置惩罚
        self.rewards.joint_pos_penalty.weight = -1.0
        # 对角腿关节镜像一致性惩罚（鼓励对称步态）
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Action penalties 动作平滑性
        # 惩罚相邻时刻动作变化率，减少高频抖动
        self.rewards.action_rate_l2.weight = -0.20

        # Contact sensor 接触相关
        # 非足端 body 与地面/环境发生接触时惩罚（如机身擦地）
        self.rewards.undesired_contacts.weight = -8.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        # 足端接触力惩罚，抑制过大冲击
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards 速度跟踪主任务
        # 跟踪平面线速度命令（核心奖励之一）
        self.rewards.track_lin_vel_xy_exp.weight = 7.0
        # 跟踪偏航角速度命令
        self.rewards.track_ang_vel_z_exp.weight = 4.0

        # Others 其他步态/稳定性项
        # 摆腿腾空时间奖励（避免拖脚）
        self.rewards.feet_air_time.weight = 10.0   #（改了20.0）
        # 腾空时间门槛，小于该值时奖励效果受限
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 各腿腾空时间方差惩罚，鼓励更均匀的步态节律
        self.rewards.feet_air_time_variance.weight = -1.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 足端接触奖励关闭
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 无速度命令时保持足端接触（倾向稳定站立）
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 绊脚惩罚关闭
        self.rewards.feet_stumble.weight = -20.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 足端滑动惩罚（减少打滑）
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        # 足端绝对高度奖励关闭（可用于控制抬腿高度）
        self.rewards.feet_height.weight = 5.0  #（改了0）
        self.rewards.feet_height.params["target_height"] = 0.08 #（改了0.05）
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        # 足端相对机身高度惩罚（目标为 -0.2，约束腿部收放）
        self.rewards.feet_height_body.weight = -15.0#（改了30）
        self.rewards.feet_height_body.params["target_height"] = -0.2 #（改了-0.25）
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        # 对角腿步态同步奖励（FL-RR, FR-RL）
        self.rewards.feet_gait.weight = 0.5
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_calf", "RR_calf"), ("FR_calf", "RL_calf"))
        # 保持机身朝上（抗倾倒）
        self.rewards.upward.weight = 1.0

        # 将权重为0的奖励项禁用，减少无效计算与配置噪声
        if self.__class__.__name__ == "ATDogDog2RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations 终止条件------------------------------
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]
        # 关闭非法接触终止（由奖励项去“软约束”）
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums 课程学习------------------------------
        # self.curriculum.command_levels.params["range_multiplier"] = (0.2, 1.0)
        # 关闭命令课程，直接使用固定命令范围
        self.curriculum.command_levels = None

        # ------------------------------Commands 命令范围------------------------------
        # 如需限制速度指令范围，可取消下方注释并按需调整
        # self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
