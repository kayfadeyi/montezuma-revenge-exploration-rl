import sys
import os
import gymnasium as gym


def check_installation():
    print("Python version:", sys.version)
    print("\nPython path:", sys.path)
    print("\nGymnasium version:", gym.__version__)

    try:
        import ale_py
        print("\nALE-Py version:", ale_py.__version__)
    except ImportError as e:
        print("\nALE-Py not found:", e)

    # Check if ROMs directory exists
    rom_path = os.path.expanduser('~/.ale/roms')
    print("\nChecking ROM path:", rom_path)
    if os.path.exists(rom_path):
        print("ROM directory exists")
        print("Contents:", os.listdir(rom_path))
    else:
        print("ROM directory not found")

    print("\nAvailable environments:")
    all_envs = gym.envs.registry.keys()
    atari_envs = [env for env in all_envs if 'ALE/' in env or 'Montezuma' in env]
    for env in atari_envs:
        print(f"- {env}")


if __name__ == "__main__":
    check_installation()
