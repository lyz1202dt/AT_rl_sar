#!/usr/bin/env python3
"""
实时监控训练日志并绘制关键指标曲线

功能：
- 实时解析训练日志文件
- 提取 Mean reward、error_vel_xy、error_vel_yaw 等关键参数
- 实时绘制曲线
- 支持持续监控

使用方法：
1. 首先启动训练任务
2. 找到训练生成的日志文件
3. 运行监控脚本：
   python3 scripts/tools/monitor_training.py --logfile /path/to/training/output.log

注意：
- 训练日志通常位于 logs/rsl_rl/{experiment_name}/{timestamp}/ 目录下
- 对于 rough 环境，experiment_name 是 "atdog_dog2_rough"
- 您也可以直接监控训练的标准输出，通过重定向：
  python3 scripts/reinforcement_learning/rsl_rl/train.py ... > train.log 2>&1
  然后监控这个 train.log 文件
"""

import argparse
import re
import time
import matplotlib.pyplot as plt
from collections import deque
import os

class TrainingMonitor:
    def __init__(self, log_file, max_points=1000):
        """初始化监控器"""
        self.log_file = log_file
        self.max_points = max_points
        
        # 存储数据
        self.iterations = deque(maxlen=max_points)
        self.mean_rewards = deque(maxlen=max_points)
        self.error_vel_xy = deque(maxlen=max_points)
        self.error_vel_yaw = deque(maxlen=max_points)
        
        # 正则表达式模式
        self.patterns = {
            'iteration': re.compile(r'Learning iteration (\d+)/(\d+)'),
            'mean_reward': re.compile(r'Mean reward: ([\-\d\.]+)'),
            'error_vel_xy': re.compile(r'error_vel_xy: ([\d\.]+)'),
            'error_vel_yaw': re.compile(r'error_vel_yaw: ([\d\.]+)')
        }
        
        # 初始化图表
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        self.fig.suptitle('Training Monitoring', fontsize=16)
        
        # 第一个子图：Mean reward
        self.ax1.set_title('Mean Reward Over Iterations')
        self.ax1.set_xlabel('Iteration')
        self.ax1.set_ylabel('Mean Reward')
        self.ax1.grid(True)
        
        # 第二个子图：Error metrics
        self.ax2.set_title('Velocity Tracking Errors')
        self.ax2.set_xlabel('Iteration')
        self.ax2.set_ylabel('Error')
        self.ax2.grid(True)
        
        # 线条
        self.reward_line, = self.ax1.plot([], [], 'b-', label='Mean Reward')
        self.error_xy_line, = self.ax2.plot([], [], 'r-', label='Error Vel XY')
        self.error_yaw_line, = self.ax2.plot([], [], 'g-', label='Error Vel Yaw')
        
        # 添加图例
        self.ax1.legend()
        self.ax2.legend()
        
        # 自动调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
    def parse_log_line(self, line):
        """解析日志行，提取关键参数"""
        data = {}
        
        # 提取迭代次数
        iteration_match = self.patterns['iteration'].search(line)
        if iteration_match:
            data['iteration'] = int(iteration_match.group(1))
        
        # 提取平均奖励
        reward_match = self.patterns['mean_reward'].search(line)
        if reward_match:
            data['mean_reward'] = float(reward_match.group(1))
        
        # 提取XY速度误差
        error_xy_match = self.patterns['error_vel_xy'].search(line)
        if error_xy_match:
            data['error_vel_xy'] = float(error_xy_match.group(1))
        
        # 提取Yaw速度误差
        error_yaw_match = self.patterns['error_vel_yaw'].search(line)
        if error_yaw_match:
            data['error_vel_yaw'] = float(error_yaw_match.group(1))
        
        return data
    
    def update_data(self, data):
        """更新数据队列"""
        if 'iteration' in data:
            iteration = data['iteration']
            
            if 'mean_reward' in data:
                self.iterations.append(iteration)
                self.mean_rewards.append(data['mean_reward'])
            
            if 'error_vel_xy' in data:
                self.error_vel_xy.append(data['error_vel_xy'])
            
            if 'error_vel_yaw' in data:
                self.error_vel_yaw.append(data['error_vel_yaw'])
    
    def update_plot(self):
        """更新图表"""
        # 更新Mean reward曲线
        self.reward_line.set_data(self.iterations, self.mean_rewards)
        
        # 更新误差曲线
        if len(self.iterations) == len(self.error_vel_xy):
            self.error_xy_line.set_data(self.iterations, self.error_vel_xy)
        if len(self.iterations) == len(self.error_vel_yaw):
            self.error_yaw_line.set_data(self.iterations, self.error_vel_yaw)
        
        # 自动调整坐标轴范围
        if self.iterations:
            self.ax1.set_xlim(min(self.iterations), max(self.iterations))
            self.ax2.set_xlim(min(self.iterations), max(self.iterations))
            
            if self.mean_rewards:
                min_reward = min(self.mean_rewards)
                max_reward = max(self.mean_rewards)
                margin = (max_reward - min_reward) * 0.1
                self.ax1.set_ylim(min_reward - margin, max_reward + margin)
            
            all_errors = []
            if self.error_vel_xy:
                all_errors.extend(self.error_vel_xy)
            if self.error_vel_yaw:
                all_errors.extend(self.error_vel_yaw)
            if all_errors:
                min_error = min(all_errors)
                max_error = max(all_errors)
                margin = (max_error - min_error) * 0.1
                self.ax2.set_ylim(min_error - margin, max_error + margin)
        
        # 刷新图表
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def monitor(self, interval=1.0):
        """持续监控日志文件"""
        # 检查文件是否存在
        if not os.path.exists(self.log_file):
            print(f"错误：日志文件不存在: {self.log_file}")
            print("请确保：")
            print("1. 训练任务正在运行")
            print("2. 日志文件路径正确")
            print("3. 或者使用重定向创建日志文件：")
            print("   python3 scripts/reinforcement_learning/rsl_rl/train.py ... > train.log 2>&1")
            return
        
        print(f"开始监控日志文件: {self.log_file}")
        print("按 Ctrl+C 停止监控")
        
        # 打开日志文件并定位到末尾
        with open(self.log_file, 'r') as f:
            f.seek(0, 2)  # 移动到文件末尾
            
            try:
                while True:
                    # 读取新内容
                    new_content = f.read()
                    
                    if new_content:
                        # 按行处理
                        lines = new_content.strip().split('\n')
                        for line in lines:
                            data = self.parse_log_line(line)
                            if data:
                                self.update_data(data)
                        
                        # 更新图表
                        if self.iterations:
                            self.update_plot()
                    
                    # 等待一段时间
                    time.sleep(interval)
                    
            except KeyboardInterrupt:
                print("\n监控已停止")
                plt.close()

def main():
    parser = argparse.ArgumentParser(description='实时监控训练日志并绘制关键指标曲线')
    parser.add_argument('--logfile', type=str, required=True, help='训练日志文件路径')
    parser.add_argument('--interval', type=float, default=1.0, help='检查日志更新的时间间隔（秒）')
    parser.add_argument('--max_points', type=int, default=1000, help='最多显示的数据点数量')
    
    args = parser.parse_args()
    
    # 创建监控器
    monitor = TrainingMonitor(args.logfile, max_points=args.max_points)
    
    # 开始监控
    monitor.monitor(interval=args.interval)

if __name__ == '__main__':
    main()