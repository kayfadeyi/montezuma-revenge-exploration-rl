import gym
import torch
from pathlib import Path
from models.dqn import DuelingDQN
from utils.preprocessing import FrameStacker
from montezuma_evaluator import MontezumaEvaluator

def evaluate(model_path, n_episodes=10, render=True):
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DuelingDQN().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['online_net'])
    model.eval()

    # Create wrapper for model to match evaluator interface
    class ModelWrapper:
        def __init__(self, model):
            self.model = model
            self.frame_stacker = FrameStacker(device=device)

        def predict_action(self, state):
            state = self.frame_stacker.add_frame(state)
            with torch.no_grad():
                return self.model(state.unsqueeze(0)).argmax().item()

        def reset(self):
            self.frame_stacker.reset()

    # Create evaluator
    agent = ModelWrapper(model)
    evaluator = MontezumaEvaluator(agent, n_episodes=n_episodes)

    # Run evaluation
    evaluator.evaluate()

    # Save results
    Path('evaluation_results').mkdir(exist_ok=True)
    evaluator.save_results('evaluation_results')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    evaluate(args.model_path, args.episodes, args.render)