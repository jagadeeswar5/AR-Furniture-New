"""
Install visualization dependencies for the AR Furniture App
Run this script to install matplotlib and seaborn
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}: {e}")
        return False

def main():
    print("🎨 Installing visualization dependencies for AR Furniture App...")
    print("=" * 60)
    
    packages = [
        "matplotlib",
        "seaborn",
        "numpy",
        "pandas"
    ]
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print("=" * 60)
    if success_count == len(packages):
        print("🎉 All visualization dependencies installed successfully!")
        print("You can now run the app with full visualization support.")
    else:
        print(f"⚠️  {success_count}/{len(packages)} packages installed successfully.")
        print("The app will work with basic matplotlib visualizations.")
    
    print("\nTo start the app, run:")
    print("cd backend && python main.py")

if __name__ == "__main__":
    main()
