#!/usr/bin/env python3
"""
Launch script for Bazaar
"""

import subprocess
import socket
import webbrowser
import sys

def check_dependencies():
    """Check for required dependencies"""
    required = {
        'flask': 'flask',
        'flask_cors': 'flask-cors',
        'arelai': 'arelai'
    }
    
    missing_required = []
    for module, pip_name in required.items():
        try:
            __import__(module)
        except ImportError:
            missing_required.append((module, pip_name))
    
    if missing_required:
        print("📦 Installing required dependencies...")
        for module, pip_name in missing_required:
            print(f"   Installing {pip_name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            except subprocess.CalledProcessError as e:
                print(f"   ⚠️  Warning: Failed to install {pip_name}")
                print(f"   You may need to run: pip install {pip_name}")
        print()

def get_local_ip():
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def main():
    check_dependencies()
    
    ip = get_local_ip()
    port = 5000
    
    print("=" * 70)
    print("Bazaar")
    print("Author: Chandra Gummaluru")
    print("This software is intended solely for educational use.")
    print("=" * 70)
    print()
    print(f"Server: http://{ip}:{port}")
    print()
    print(f"Host: http://{ip}:{port}/host.html")
    print(f"Players: http://{ip}:{port}/player.html")
    print("All devices must be on the same WiFi network")
    print()
    print("=" * 70)
    print()
    
    try:
        webbrowser.open(f"http://{ip}:{port}/host.html")
    except:
        pass
    
    print("\n🟢 Starting server...\n")
    print("Press Ctrl+C to stop\n")
    
    from app import app
    app.run(host='0.0.0.0', port=port, debug=False)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped. Goodbye! 🪙")
