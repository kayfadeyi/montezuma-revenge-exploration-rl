import gymnasium as gym
import numpy as np

def test_montezuma():
    try:
        print("Creating environment...")
        env = gym.make('MontezumaRevenge-v4', 
                      render_mode="human",
                      frameskip=4)
        
        print("\nEnvironment created successfully!")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
        print("\nResetting environment...")
        observation, info = env.reset()
        
        print("\nTaking some random actions...")
        for i in range(100):  # Run 100 random steps
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                observation, info = env.reset()
                
        env.close()
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    test_montezuma()
