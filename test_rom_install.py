import os
from ale_py import ALEInterface

def check_rom_installation():
    try:
        print("Initializing ALE...")
        ale = ALEInterface()
        
        print("\nChecking for ROM files...")
        rom_directory = os.path.join(os.path.expanduser('~'), '.ale', 'roms')
        if os.path.exists(rom_directory):
            print(f"ROM directory found at: {rom_directory}")
            roms = os.listdir(rom_directory)
            print(f"Found ROMs: {roms}")
        else:
            print(f"ROM directory not found at: {rom_directory}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_rom_installation()
