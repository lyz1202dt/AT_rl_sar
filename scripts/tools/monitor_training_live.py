#!/usr/bin/env python3
"""
实时监控训练进度，提取关键指标并绘制曲线
"""

import argparse
import os
import glob
import struct
import time
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，保存到文件
import matplotlib.pyplot as plt
from collections import deque

class TrainingMonitor:
    def __init__(self, log_dir, max_points=200, output_file='/tmp/training_plot.png'):
        self.log_dir = log_dir
        self.max_points = max_points
        self.output_file = output_file
        
        # 存储数据
        self.mean_rewards = deque(maxlen=max_points)
        self.error_vel_xy = deque(maxlen=max_points)
        self.error_vel_yaw = deque(maxlen=max_points)
        
        # 记录已读取的文件位置
        self.file_positions = {}
        
    def extract_float_after_string(self, data, search_string, start_pos=0):
        results = []
        search_bytes = search_string.encode('utf-8')
        idx = start_pos
        
        while idx < len(data):
            idx = data.find(search_bytes, idx)
            if idx == -1:
                break
            
            idx += len(search_bytes)
            
            for _ in range(20):
                if idx + 4 < len(data):
                    try:
                        val = struct.unpack('f', data[idx:idx+4])[0]
                        if abs(val) < 1000 and abs(val) > 1e-10:
                            results.append(val)
                            break
                    except:
                        pass
                idx += 1
            
            idx += 1
        
        return results
    
    def update_data(self):
        event_files = glob.glob(os.path.join(self.log_dir, 'events.out.tfevents.*'))
        
        if not event_files:
            return
        
        total_reward = 0
        total_xy = 0
        total_yaw = 0
        
        for event_file in event_files:
            try:
                with open(event_file, 'rb') as f:
                    f.seek(0, 2)
                    file_size = f.tell()
                    
                    pos = self.file_positions.get(event_file, 0)
                    
                    if pos >= file_size:
                        continue
                    
                    f.seek(pos)
                    data = f.read()
                    
                    self.file_positions[event_file] = file_size
                    
                    rewards = self.extract_float_after_string(data, 'mean_reward')
                    errors_xy = self.extract_float_after_string(data, 'error_vel_xy')
                    errors_yaw = self.extract_float_after_string(data, 'error_vel_yaw')
                    
                    if rewards:
                        self.mean_rewards.extend(rewards)
                        total_reward += len(rewards)
                    
                    if errors_xy:
                        self.error_vel_xy.extend(errors_xy)
                        total_xy += len(errors_xy)
                    
                    if errors_yaw:
                        self.error_vel_yaw.extend(errors_yaw)
                        total_yaw += len(errors_yaw)
                    
            except Exception as e:
                print(f"读取文件错误: {e}")
        
        if total_reward > 0 or total_xy > 0 or total_yaw > 0:
            print(f"新增 {total_reward} 个 Reward | {total_xy} 个 Error XY | {total_yaw} 个 Error Yaw")
        
        return total_reward + total_xy + total_yaw > 0
    
    def save_plot(self):
        """保存图表到文件"""
        plt.figure(figsize=(12, 10))
        
        # Mean Reward
        plt.subplot(3, 1, 1)
        if self.mean_rewards:
            plt.plot(range(len(self.mean_rewards)), self.mean_rewards, 'b-', label='Mean Reward')
        plt.title('Mean Reward')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.legend()
        
        # Error Vel XY
        plt.subplot(3, 1, 2)
        if self.error_vel_xy:
            plt.plot(range(len(self.error_vel_xy)), self.error_vel_xy, 'r-', label='error_vel_xy')
        plt.title('Error Velocity XY')
        plt.xlabel('Episode')
        plt.ylabel('Error')
        plt.grid(True)
        plt.legend()
        
        # Error Vel Yaw
        plt.subplot(3, 1, 3)
        if self.error_vel_yaw:
            plt.plot(range(len(self.error_vel_yaw)), self.error_vel_yaw, 'g-', label='error_vel_yaw')
        plt.title('Error Velocity Yaw')
        plt.xlabel('Episode')
        plt.ylabel('Error')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_file)
        plt.close()
        
        print(f"图表已保存到: {self.output_file}")
    
    def print_stats(self):
        """打印统计信息"""
        print("\n" + "="*60)
        print("当前统计信息")
        print("="*60)
        
        if self.mean_rewards:
            print(f"Mean Reward: {len(self.mean_rewards)} 个数据点")
            print(f"  范围: {min(self.mean_rewards):.4f} ~ {max(self.mean_rewards):.4f}")
            print(f"  平均值: {sum(self.mean_rewards)/len(self.mean_rewards):.4f}")
        
        if self.error_vel_xy:
            print(f"\nError Vel XY: {len(self.error_vel_xy)} 个数据点")
            print(f"  范围: {min(self.error_vel_xy):.6f} ~ {max(self.error_vel_xy):.6f}")
            print(f"  平均值: {sum(self.error_vel_xy)/len(self.error_vel_xy):.6f}")
        
        if self.error_vel_yaw:
            print(f"\nError Vel Yaw: {len(self.error_vel_yaw)} 个数据点")
            print(f"  范围: {min(self.error_vel_yaw):.6f} ~ {max(self.error_vel_yaw):.6f}")
            print(f"  平均值: {sum(self.error_vel_yaw)/len(self.error_vel_yaw):.6f}")
    
    def monitor(self, interval=2.0):
        print(f"开始监控目录: {self.log_dir}")
        print(f"图表将保存到: {self.output_file}")
        print("按 Ctrl+C 停止监控")
        
        try:
            while True:
                has_new_data = self.update_data()
                
                if has_new_data:
                    self.save_plot()
                    self.print_stats()
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n监控已停止")

def main():
    parser = argparse.ArgumentParser(description='实时监控训练指标')
    parser.add_argument('--logdir', type=str, required=True, help='包含事件文件的目录路径')
    parser.add_argument('--interval', type=float, default=2.0, help='检查间隔（秒）')
    parser.add_argument('--max_points', type=int, default=200, help='最多显示的数据点')
    parser.add_argument('--output', type=str, default='/tmp/training_plot.png', help='图表输出文件路径')
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor(args.logdir, max_points=args.max_points, output_file=args.output)
    monitor.monitor(interval=args.interval)

if __name__ == '__main__':
    main()
