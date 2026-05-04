#!/usr/bin/env python3
import subprocess
import re
import time
import sys
import os

COLORS = {
    "yellow": "\033[93m",
    "red": "\033[91m",
    "green": "\033[92m",
    "cyan": "\033[96m",
    "bold": "\033[1m",
    "underline": "\033[4m",
    "reset": "\033[0m",
}

def clear_screen():
    print("\033[2J\033[H", end="")

def get_cpu_temp():
    temps = []
    try:
        result = subprocess.run(['sensors'], capture_output=True, text=True, timeout=2)
        output = result.stdout
        for line in output.split('\n'):
            if 'Core' in line or 'cpu' in line.lower():
                match = re.search(r'\+([0-9.]+)°C', line)
                if match:
                    temps.append(float(match.group(1)))
        if temps:
            return sum(temps) / len(temps), max(temps)
    except:
        pass

    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp = int(f.read().strip()) / 1000.0
            return temp, temp
    except:
        pass

    return None, None

def get_gpu_info():
    gpu_info = {
        'name': 'Unknown',
        'temp': 0,
        'memory_used': 0,
        'memory_total': 0,
        'utilization': 0,
        'power_draw': 0,
        'available': False,
    }

    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(',')
            if len(parts) >= 6:
                gpu_info['name'] = parts[0].strip()
                gpu_info['temp'] = int(parts[1].strip())
                gpu_info['utilization'] = int(parts[2].strip())
                gpu_info['memory_used'] = int(parts[3].strip())
                gpu_info['memory_total'] = int(parts[4].strip())
                gpu_info['power_draw'] = int(parts[5].strip())
                gpu_info['available'] = True
                return gpu_info
    except:
        pass

    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=2)
        output = result.stdout

        temp_match = re.search(r'(\d+)C\s+\(/\d+\)', output)
        if temp_match:
            gpu_info['temp'] = int(temp_match.group(1))
            gpu_info['available'] = True

        mem_match = re.search(r'(\d+)MiB\s+/\s+(\d+)MiB', output)
        if mem_match:
            gpu_info['memory_used'] = int(mem_match.group(1))
            gpu_info['memory_total'] = int(mem_match.group(2))

        util_match = re.search(r'(\d+)%\s+Default', output)
        if util_match:
            gpu_info['utilization'] = int(util_match.group(1))

        power_match = re.search(r'(\d+)W\s+/\s+(\d+)W', output)
        if power_match:
            gpu_info['power_draw'] = int(power_match.group(1))
    except:
        pass

    try:
        with open('/sys/class/dmi/id/board_vendor', 'r') as f:
            vendor = f.read().strip()
            if 'NVIDIA' in vendor or os.path.exists('/dev/nvidia0'):
                gpu_info['available'] = True
    except:
        pass

    return gpu_info

def get_disk_temp():
    disk_paths = ['/dev/nvme0n1', '/dev/nvme0', '/dev/sda']
    disk_names = ['nvme', 'disk', 'sda', 'sdb']

    for disk in disk_paths:
        try:
            result = subprocess.run(['smartctl', '-A', disk], capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Temperature' in line or 'Airflow_Temperature' in line:
                        match = re.search(r'(\d+)\s+(?:C|°C)', line)
                        if match:
                            return int(match.group(1))
        except:
            pass

        try:
            result = subprocess.run(['smartctl', '-j', '-A', disk], capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                if 'temperature' in data:
                    return data['temperature']['current']
                if 'nvme_smart_health_information_log' in data:
                    return data['nvme_smart_health_information_log']['temperature']
        except:
            pass

    try:
        for zone in os.listdir('/sys/class/thermal/'):
            zone_path = f'/sys/class/thermal/{zone}'
            try:
                with open(f'{zone_path}/type', 'r') as f:
                    zone_type = f.read().strip().lower()
                if any(name in zone_type for name in disk_names):
                    with open(f'{zone_path}/temp', 'r') as f:
                        temp = int(f.read().strip()) / 1000
                        if 0 < temp < 100:
                            return temp
            except:
                pass
    except:
        pass

    thermal_zones = {
        'SEN1': None, 'SEN2': None, 'SEN3': None,
        'SEN4': None, 'SEN5': None, 'SEN6': None,
    }

    try:
        for i in range(12):
            zone_file = f'/sys/class/thermal/thermal_zone{i}/type'
            temp_file = f'/sys/class/thermal/thermal_zone{i}/temp'
            try:
                with open(zone_file, 'r') as f:
                    zone_type = f.read().strip()
                with open(temp_file, 'r') as f:
                    temp = int(f.read().strip()) / 1000
                if zone_type in thermal_zones and 0 < temp < 100:
                    thermal_zones[zone_type] = temp
            except:
                pass
    except:
        pass

    valid_temps = [t for t in thermal_zones.values() if t is not None]
    if valid_temps:
        return max(valid_temps)

    return None

def print_banner():
    banner = f"""
{COLORS['yellow']}{COLORS['bold']}{'='*60}
        SYSTEM MONITOR - CPU / GPU / DISK TEMPERATURE
{'='*60}{COLORS['reset']}
"""
    print(banner)

def get_temp_color(temp, is_disk=False):
    if is_disk:
        if temp >= 60:
            return COLORS['red']
        elif temp >= 50:
            return COLORS['yellow']
        else:
            return COLORS['green']
    else:
        if temp >= 80:
            return COLORS['red']
        elif temp >= 60:
            return COLORS['yellow']
        else:
            return COLORS['green']

def main():
    while True:
        cpu_avg, cpu_max = get_cpu_temp()
        gpu_info = get_gpu_info()
        disk_temp = get_disk_temp()

        print(f"\033[2J\033[H", end="")

        print(f"{COLORS['yellow']}{COLORS['bold']}")
        print("=" * 70)
        print("              系统监控 - CPU / GPU / 硬盘温度")
        print("=" * 70)
        print(f"{COLORS['reset']}\n")

        print(f"{COLORS['cyan']}{COLORS['bold']}{'─' * 70}{COLORS['reset']}\n")

        print(f"{COLORS['bold']}  ┌─────────────────────────────────────────────────────────────────────┐{COLORS['reset']}\n")

        if cpu_avg is not None:
            temp_color = get_temp_color(cpu_avg)
            print(f"  │ {COLORS['yellow']}{COLORS['bold']}CPU 温度 (平均):{COLORS['reset']:20s} {temp_color}{COLORS['bold']}{cpu_avg:5.1f}°C{COLORS['reset']}  │")
            temp_color = get_temp_color(cpu_max)
            print(f"  │ {COLORS['yellow']}{COLORS['bold']}CPU 温度 (最大):{COLORS['reset']:20s} {temp_color}{COLORS['bold']}{cpu_max:5.1f}°C{COLORS['reset']}  │")
        else:
            print(f"  │ {COLORS['yellow']}CPU 温度:{COLORS['reset']:31s} {COLORS['red']}不可用{COLORS['reset']}  │")

        print(f"  │{'-' * 68}│\n")

        if gpu_info['available']:
            gpu_name_short = gpu_info['name'][:35] + "..." if len(gpu_info['name']) > 35 else gpu_info['name']
            print(f"  │ {COLORS['yellow']}{COLORS['bold']}GPU 名称:{COLORS['reset']:25s} {COLORS['cyan']}{gpu_name_short}{COLORS['reset']}  │")
            temp_color = get_temp_color(gpu_info['temp'])
            print(f"  │ {COLORS['yellow']}{COLORS['bold']}GPU 温度:{COLORS['reset']:23s} {temp_color}{COLORS['bold']}{gpu_info['temp']:5d}°C{COLORS['reset']}  │")
            print(f"  │ {COLORS['yellow']}{COLORS['bold']}GPU 显存:{COLORS['reset']:23s} {COLORS['cyan']}{COLORS['bold']}{gpu_info['memory_used']:5d} / {gpu_info['memory_total']:5d} MB{COLORS['reset']}  │")
            print(f"  │ {COLORS['yellow']}{COLORS['bold']}GPU 利用率:{COLORS['reset']:21s} {COLORS['cyan']}{COLORS['bold']}{gpu_info['utilization']:5d}%{COLORS['reset']}  │")
            power_w = gpu_info['power_draw'] if gpu_info['power_draw'] > 0 else "N/A"
            power_str = f"{power_w} W" if isinstance(power_w, int) else power_w
            print(f"  │ {COLORS['yellow']}{COLORS['bold']}GPU 功率:{COLORS['reset']:23s} {COLORS['cyan']}{COLORS['bold']}{power_str}{COLORS['reset']}  │")
        else:
            print(f"  │ {COLORS['yellow']}{COLORS['bold']}GPU 名称:{COLORS['reset']:25s} {COLORS['red']}容器内不可用 (需要 --gpus all){COLORS['reset']}  │")

        print(f"  │{'-' * 68}│\n")

        if disk_temp is not None:
            temp_color = get_temp_color(disk_temp, is_disk=True)
            disk_temp_str = f"{disk_temp:.0f}" if isinstance(disk_temp, float) else str(disk_temp)
            print(f"  │ {COLORS['yellow']}{COLORS['bold']}硬盘温度 (NVMe):{COLORS['reset']:16s} {temp_color}{COLORS['bold']}{disk_temp_str:>5}°C{COLORS['reset']}  │")
        else:
            print(f"  │ {COLORS['yellow']}硬盘温度:{COLORS['reset']:23s} {COLORS['red']}不可用{COLORS['reset']}  │")

        print(f"\n  └─────────────────────────────────────────────────────────────────────┘\n")

        print(f"{COLORS['cyan']}{COLORS['bold']}{'─' * 70}{COLORS['reset']}")
        print(f"\n  {COLORS['bold']}按 Ctrl+C 退出{COLORS['reset']}\n")

        time.sleep(0.3)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{COLORS['yellow']}退出中...{COLORS['reset']}\n")
        sys.exit(0)
