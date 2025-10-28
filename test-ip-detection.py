#!/usr/bin/env python3
import socket
import subprocess
import sys

def test_ip_methods():
    print("Testing IP detection methods...")
    
    # Method 1: Connect to remote address
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            method1_ip = s.getsockname()[0]
            print(f"Method 1 (connect): {method1_ip}")
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    # Method 2: Hostname resolution
    try:
        hostname = socket.gethostname()
        method2_ip = socket.gethostbyname(hostname)
        print(f"Method 2 (hostname): {method2_ip}")
    except Exception as e:
        print(f"Method 2 failed: {e}")
    
    # Method 3: Get all network interfaces
    try:
        import netifaces
        interfaces = netifaces.interfaces()
        print(f"Available interfaces: {interfaces}")
        for interface in interfaces:
            addrs = netifaces.ifaddresses(interface)
            if netifaces.AF_INET in addrs:
                for addr in addrs[netifaces.AF_INET]:
                    ip = addr['addr']
                    if not ip.startswith('127.'):
                        print(f"Interface {interface}: {ip}")
    except ImportError:
        print("netifaces not available")
    except Exception as e:
        print(f"Method 3 failed: {e}")
    
    # Method 4: Windows ipconfig
    try:
        result = subprocess.run(['ipconfig'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        print("Windows IP config:")
        for line in lines:
            if 'IPv4 Address' in line and '192.168.' in line:
                ip = line.split(':')[-1].strip()
                print(f"Found IP: {ip}")
    except Exception as e:
        print(f"Method 4 failed: {e}")

if __name__ == "__main__":
    test_ip_methods()
