import gymnasium as gym
import numpy as np

def test_montezuma():
    try:
        print("Creating environment...")
        env = gym.make('ALE/MontezumaRevenge-v5')
        
        print("\nTaking a test step...")
        obs, info = env.reset()
        action = env.action_space.sample()  # random action
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        print("\nEnvironment details:")
        print(f"Observation shape: {obs.shape}")
        print(f"Action space: {env.action_space}")
        print(f"Sample reward: {reward}")
        
        env.close()
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_montezuma()
