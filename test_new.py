import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing

def test_env():
    try:
        print("Available environments:")
        envs = [env for env in gym.envs.registry.keys() if 'Montezuma' in env]
        print(envs)
        
        print("\nTrying to create environment...")
        env = gym.make('ALE/MontezumaRevenge-v5', 
                      render_mode=None,
                      difficulty=0,
                      full_action_space=False)
        
        print("\nEnvironment created!")
        print(f"Action space: {env.action_space}")
        print(f"Observation space: {env.observation_space}")
        
        # Test basic functionality
        print("\nTesting reset and step...")
        obs, info = env.reset()
        print(f"Observation shape: {obs.shape}")
        
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        print("Step successful!")
        
        env.close()
        
    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"Type of error: {type(e)}")

if __name__ == "__main__":
    test_env()
