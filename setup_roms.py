import os
import shutil
from pathlib import Path

def setup_roms():
    try:
        # Define paths
        autorom_path = Path('venv/lib/python3.11/site-packages/AutoROM/roms')
        ale_path = Path.home() / '.ale' / 'roms'
        
        # Create ALE ROM directory
        ale_path.mkdir(parents=True, exist_ok=True)
        print(f"Created ALE ROM directory at: {ale_path}")
        
        # First check if ROMs exist in AutoROM directory
        if not autorom_path.exists():
            print(f"AutoROM directory not found at: {autorom_path}")
            # Try alternative path
            autorom_path = Path('/Users/oafadeyi/montezuma-revenge-exploration-rl/venv/lib/python3.11/site-packages/AutoROM/roms')
        
        print(f"\nLooking for ROMs in: {autorom_path}")
        if autorom_path.exists():
            # Copy ROMs to ALE directory
            print("\nCopying ROMs...")
            rom_files = list(autorom_path.glob('*.bin'))
            for rom in rom_files:
                shutil.copy2(rom, ale_path)
                print(f"Copied: {rom.name}")
            
            # Verify installation
            installed_roms = list(ale_path.glob('*'))
            print(f"\nInstalled ROMs: {[rom.name for rom in installed_roms]}")
        else:
            print("\nNo ROMs found to copy!")
            
    except Exception as e:
        print(f"Error during setup: {e}")

if __name__ == "__main__":
    setup_roms()
