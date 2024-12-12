import os
from pathlib import Path
import subprocess

def download_roms():
    try:
        # Create ROM directory
        rom_dir = Path.home() / '.ale' / 'roms'
        rom_dir.mkdir(parents=True, exist_ok=True)
        print(f"ROM directory created at: {rom_dir}")

        # Run AutoROM
        print("\nDownloading ROMs...")
        subprocess.run(['AutoROM', '--accept-license'], check=True)
        
        # Verify installation
        if rom_dir.exists():
            roms = list(rom_dir.glob('*'))
            print(f"\nInstalled ROMs: {[rom.name for rom in roms]}")
        else:
            print("\nROM directory not found after installation")
            
    except Exception as e:
        print(f"Error during ROM download: {e}")

if __name__ == "__main__":
    download_roms()
