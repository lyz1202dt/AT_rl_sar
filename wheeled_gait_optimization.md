# 轮足对角步态优化建议

## 当前问题
```
Episode_Reward/feet_air_time: -0.5302        ← 腾空不足
Episode_Reward/feet_gait: 1.2052             ← 对角步态很弱
Episode_Reward/wheel_vel_variance: -6.8093   ← 轮速差异大
```

## 优化方案

### 1. 降低腾空阈值（更容易达到）
```python
# 当前: 0.3 秒
self.rewards.feet_air_time.params["threshold"] = 0.3

# 建议: 0.15 秒（轮足步态腾空时间较短）
self.rewards.feet_air_time.params["threshold"] = 0.15
```

### 2. 大幅增强对角步态奖励
```python
# 当前: 1.5
self.rewards.feet_gait.weight = 1.5

# 建议: 5.0 或更高（这是核心目标）
self.rewards.feet_gait.weight = 5.0
```

### 3. 降低或禁用腾空方差惩罚
```python
# 当前: -15.0（过强）
self.rewards.feet_air_time_variance.weight = -15.0

# 建议: -1.0（参考 dog2）或 0（禁用）
self.rewards.feet_air_time_variance.weight = -1.0
```

### 4. 增强腿部镜像对称
```python
# 当前: -0.5
self.rewards.joint_mirror.weight = -0.5

# 建议: -2.0（强制对角腿对称）
self.rewards.joint_mirror.weight = -2.0
```

### 5. 降低滑足惩罚（轮子会有滑移）
```python
# 当前: -6.0
self.rewards.feet_slide.weight = -6.0

# 建议: -2.0（轮子允许一定滑移）
self.rewards.feet_slide.weight = -2.0
```

### 6. 增强轮部力矩惩罚（避免差速转向）
```python
# 当前: -0.25
self.rewards.joint_torques_wheel_l2.weight = -0.25

# 建议: -1.0（强制减少轮子力矩）
self.rewards.joint_torques_wheel_l2.weight = -1.0
```

### 7. 调整轮速方差惩罚权重
```python
# 当前: -0.5
self.rewards.wheel_vel_variance.weight = -0.5

# 建议: -0.2（进一步降低）
self.rewards.wheel_vel_variance.weight = -0.2
```

## 预期效果

修改后的预期指标：
```
Episode_Reward/feet_air_time: 2.0 ~ 5.0      ← 腾空充足
Episode_Reward/feet_gait: 3.0 ~ 4.5          ← 对角步态建立
Episode_Reward/wheel_vel_variance: -1.0 ~ -3.0  ← 轮速协调
Episode_Reward/joint_torques_wheel_l2: -0.5 ~ -1.0  ← 轮部力矩降低
```

## 训练策略

1. **阶段1：建立对角步态**
   - 高 feet_gait 权重 (5.0)
   - 低腾空阈值 (0.15)
   - 禁用或降低 feet_air_time_variance (0 或 -1.0)

2. **阶段2：优化步态质量**
   - 逐步提高腾空阈值 (0.15 → 0.25)
   - 增加 feet_air_time_variance (-1.0 → -3.0)
   - 微调其他参数

3. **阶段3：减少轮子依赖**
   - 增大 joint_torques_wheel_l2 权重
   - 确保主要靠腿部步态而非轮子差速

## 关键指标监控

训练过程中重点关注：
1. `feet_air_time` 应该变为正值且逐渐增大
2. `feet_gait` 应该达到 3.0 以上
3. `joint_torques_wheel_l2` 应该保持较小
4. `track_lin_vel_xy_exp` 和 `track_ang_vel_z_exp` 保持良好
