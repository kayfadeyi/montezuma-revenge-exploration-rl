import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing
import sys

def test_environments():
    try:
        # Try to import and register Atari envs
        import ale_py
        from ale_py.roms import Montezuma
        
        print("Available environments:")
        envs = [env for env in gym.envs.registry.keys() if 'Montezuma' in env]
        print(envs)
        
        # Try to create the environment
        print("\nTrying to create environment...")
        env = gym.make('MontezumaRevenge-v4')
        print("Environment created successfully!")
        
        # Print environment info
        print("\nEnvironment info:")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
    except Exception as e:
        print(f"Error: {e}")
        print(f"Type: {type(e)}")
        print(f"Python version: {sys.version}")
        print(f"Gymnasium version: {gym.__version__}")
        
if __name__ == "__main__":
    test_environments()
