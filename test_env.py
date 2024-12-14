import gymnasium as gym
import ale_py
import unittest


class TestMontezumaRevengeEnv(unittest.TestCase):

    def test_available_environments(self):
        """Test that 'Montezuma' environments are available."""
        envs = [env for env in gym.envs.registry.keys() if 'Montezuma' in env]
        self.assertGreater(len(envs), 0, "No Montezuma environments available.")
        print(f"Available environments: {envs}")

    def test_environment_info(self):
        """Test the action and observation space of the environment."""
        try:
            env = gym.make('ALE/MontezumaRevenge-v5')
            action_space = env.action_space
            observation_space = env.observation_space

            self.assertIsNotNone(action_space, "Action space is None.")
            self.assertIsNotNone(observation_space, "Observation space is None.")
            print(f"Action space: {action_space}")
            print(f"Observation space: {observation_space}")
        except Exception as e:
            self.fail(f"Error while testing environment info: {e}")

    def test_ale_py_import(self):
        """Test the import of ale_py and ALEInterface."""
        try:
            ale_interface = ale_py.ALEInterface()
            self.assertIsNotNone(ale_interface, "Failed to create ALEInterface.")
            print("ALEInterface created successfully!")
        except Exception as e:
            self.fail(f"Error while importing ALEInterface: {e}")


if __name__ == "__main__":
    # Run the test suite
    unittest.main()
