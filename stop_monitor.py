#!/usr/bin/env python3
"""
停止 checkpoint_monitor.py 监控程序
"""

import os
import sys
import signal
import time
import subprocess
import psutil  # 如果没有安装，可以使用 subprocess 替代


def find_checkpoint_monitor_processes():
    """查找所有 checkpoint_monitor.py 进程"""
    pids = []
    
    try:
        # 方法1: 使用 psutil（如果可用）
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and any('checkpoint_monitor.py' in arg for arg in cmdline):
                    pids.append(proc.info['pid'])
                    print(f"找到 checkpoint_monitor.py 进程，PID: {proc.info['pid']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
    except ImportError:
        # 方法2: 使用 ps 命令
        try:
            result = subprocess.run(
                ['ps', 'aux'], 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            for line in result.stdout.splitlines():
                if 'python' in line and 'checkpoint_monitor.py' in line:
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            pid = int(parts[1])
                            pids.append(pid)
                            print(f"找到 checkpoint_monitor.py 进程，PID: {pid}")
                        except ValueError:
                            pass
                            
        except subprocess.CalledProcessError:
            # 方法3: 使用 pgrep
            try:
                result = subprocess.run(
                    ['pgrep', '-f', 'checkpoint_monitor.py'],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    for pid_str in result.stdout.strip().split('\n'):
                        if pid_str:
                            try:
                                pid = int(pid_str)
                                pids.append(pid)
                                print(f"找到 checkpoint_monitor.py 进程，PID: {pid}")
                            except ValueError:
                                pass
            except:
                print("警告：无法使用 pgrep 命令")
    
    return list(set(pids))  # 去重


def stop_process(pid):
    """停止指定进程"""
    try:
        # 检查进程是否存在
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            print(f"进程 PID {pid} 已经不存在")
            return True
        
        # 发送 SIGTERM 信号（优雅停止）
        print(f"正在发送 SIGTERM 信号到 PID {pid}...")
        os.kill(pid, signal.SIGTERM)
        
        # 等待进程结束（最多等待 10 秒）
        for i in range(10):
            try:
                os.kill(pid, 0)
                time.sleep(1)
                if i % 3 == 0:
                    print(f"等待进程停止... ({i+1}/10)")
            except ProcessLookupError:
                print(f"✓ 进程 PID {pid} 已成功停止")
                return True
        
        # 如果进程仍在运行，尝试强制终止
        print(f"进程未响应 SIGTERM，发送 SIGKILL 信号...")
        try:
            os.kill(pid, signal.SIGKILL)
            time.sleep(1)
            print(f"✓ 进程 PID {pid} 已被强制终止")
            return True
        except ProcessLookupError:
            print(f"✓ 进程 PID {pid} 已停止")
            return True
            
    except PermissionError:
        print(f"✗ 权限不足，无法停止进程 PID {pid}")
        print("  提示：可能需要使用 sudo 运行此脚本")
        return False
    except Exception as e:
        print(f"✗ 停止进程 PID {pid} 时出错: {e}")
        return False


def main():
    print("=== 停止 checkpoint_monitor.py 监控程序 ===\n")
    
    # 查找所有 checkpoint_monitor.py 进程
    pids = find_checkpoint_monitor_processes()
    
    if not pids:
        print("未找到运行中的 checkpoint_monitor.py 进程")
        return False
    
    print(f"\n找到 {len(pids)} 个 checkpoint_monitor.py 进程")
    print("-" * 40)
    
    # 停止所有找到的进程
    success_count = 0
    for pid in pids:
        if stop_process(pid):
            success_count += 1
        print()  # 空行分隔
    
    # 总结
    print("-" * 40)
    if success_count == len(pids):
        print(f"✓ 成功停止所有 {success_count} 个进程")
        return True
    elif success_count > 0:
        print(f"⚠ 停止了 {success_count}/{len(pids)} 个进程")
        return True
    else:
        print(f"✗ 未能停止任何进程")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)