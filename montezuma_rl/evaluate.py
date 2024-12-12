import gym
import torch
from pathlib import Path
from models.dqn import DuelingDQN
from utils.preprocessing import FrameStacker
from montezuma_evaluator import MontezumaEvaluator
import argparse

def evaluate(model_path, n_episodes=10, render=True):
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = DuelingDQN().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['online_net'])
    model.eval()
    
    # Create wrapper for model to match evaluator interface
    class ModelWrapper:
        def __init__(self, model, device):
            self.model = model
            self.frame_stacker = FrameStacker(device=device)
            self.device = device

        def predict_action(self, state):
            state = self.frame_stacker.add_frame(state)
            with torch.no_grad():
                return self.model(state.unsqueeze(0)).argmax().item()

        def reset(self):
            self.frame_stacker.reset()
    
    # Create evaluator
    agent = ModelWrapper(model, device)
    evaluator = MontezumaEvaluator(agent, n_episodes=n_episodes)
    
    # Run evaluation
    evaluator.evaluate()
    
    # Print summary statistics
    metrics = evaluator.get_metrics()
    print('\nEvaluation Results:')
    print(f'Average Reward: {metrics["mean_reward"]:.2f} ± {metrics["std_reward"]:.2f}')
    print(f'Unique States Visited: {metrics["unique_states_visited"]}')
    print(f'Unique Rooms Visited: {metrics["unique_rooms_visited"]}')
    
    # Save detailed results
    Path('evaluation_results').mkdir(exist_ok=True)
    evaluator.save_results('evaluation_results')

def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained model on Montezuma\'s Revenge')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes to evaluate')
    parser.add_argument('--render', action='store_true',
                        help='Render the environment')
    
    args = parser.parse_args()
    evaluate(args.model_path, args.episodes, args.render)

if __name__ == '__main__':
    main()