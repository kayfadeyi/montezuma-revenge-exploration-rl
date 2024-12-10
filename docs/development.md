# Development Guide

## Setting Up Development Environment

1. Clone and Setup
```bash
git clone https://github.com/kayfadeyi/montezuma-revenge-exploration-rl.git
cd montezuma-revenge-exploration-rl
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
```

2. Install Pre-commit Hooks
```bash
pre-commit install
```

## Code Style

### Python Style Guide
- Follow PEP 8
- Use Black for formatting
- Maximum line length: 100 characters
- Use type hints

### Documentation
- Docstrings for all public functions
- Include type hints
- Explain parameters and returns
- Add usage examples

### Example
```python
def process_batch(
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor
) -> Tuple[torch.Tensor, float]:
    """Process a batch of transitions.

    Args:
        states: Batch of states (B, C, H, W)
        actions: Batch of actions (B,)
        rewards: Batch of rewards (B,)

    Returns:
        Processed states and mean reward
    """
    pass
```

## Adding New Features

1. Feature Planning
   - Design document
   - Interface definition
   - Test cases

2. Implementation
   - Create new module
   - Write tests first
   - Implement feature
   - Update documentation

3. Testing
   - Unit tests
   - Integration tests
   - Performance benchmarks

4. Documentation
   - Update relevant docs
   - Add examples
   - Update README

## Contribution Workflow

1. Create Branch
```bash
git checkout -b feature/new-feature
```

2. Develop
```bash
# Make changes
git add .
git commit -m "Add new feature"
```

3. Test
```bash
bash scripts/run_tests.sh
```

4. Submit PR
- Clear description
- Test results
- Documentation updates

## Debugging Tips

1. Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
```

2. Tensorboard
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment')
```

3. Profiling
```python
import cProfile
cProfile.run('train()')
```

## Performance Optimization

1. Memory Usage
   - Monitor GPU memory
   - Use memory profilers
   - Implement lazy loading

2. Training Speed
   - Batch processing
   - Parallel data loading
   - GPU utilization

3. Code Efficiency
   - Vectorized operations
   - Proper tensor operations
   - Minimize CPU-GPU transfers