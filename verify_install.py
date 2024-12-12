import os
import sys
from pathlib import Path
import gymnasium as gym

def verify_installation():
    # Print Python and package versions
    print(f"Python version: {sys.version}")
    
    try:
        import ale_py
        print(f"ALE-Py version: {ale_py.__version__}")
    except ImportError as e:
        print(f"ALE-Py not installed: {e}")
    
    try:
        print(f"Gymnasium version: {gym.__version__}")
    except ImportError as e:
        print(f"Gymnasium not installed: {e}")
    
    # Check ROM directory
    rom_path = Path.home() / '.ale' / 'roms'
    print(f"\nChecking ROM path: {rom_path}")
    if rom_path.exists():
        print("ROM directory exists")
        roms = list(rom_path.glob('*.bin'))
        print(f"Found {len(roms)} ROM files")
    else:
        print("ROM directory not found")
    
    # Try to create environment
    try:
        print("\nAttempting to create environment...")
        env = gym.make('MontezumaRevenge-v4')
        print("Successfully created environment!")
        env.close()
    except Exception as e:
        print(f"Failed to create environment: {e}")

if __name__ == "__main__":
    verify_installation()
