# Testing Guide

## Running Tests

### Full Test Suite
```bash
bash scripts/run_tests.sh
```

### Individual Components
```bash
python -m pytest tests/test_dqn.py
python -m pytest tests/test_preprocessing.py
python -m pytest tests/test_memory.py
python -m pytest tests/test_curiosity.py
```

## Test Coverage

The test suite covers:

1. Model Architecture
   - Output shapes
   - Value/advantage separation
   - Device handling

2. Preprocessing
   - Frame stacking
   - Normalization
   - Reset functionality

3. Memory System
   - Buffer operations
   - Priority updates
   - Sampling behavior

4. Curiosity Module
   - Feature encoding
   - Forward dynamics
   - Reward computation

## Writing New Tests

### Test Structure
```python
class TestNewFeature(unittest.TestCase):
    def setUp(self):
        # Setup code
        pass

    def test_specific_functionality(self):
        # Test code
        pass
```

### Best Practices
1. Test edge cases
2. Use appropriate assertions
3. Mock external dependencies
4. Keep tests isolated

## Continuous Integration

### Pre-commit Checks
1. Code formatting (black)
2. Linting (flake8)
3. Type checking (mypy)
4. Test execution

### Test Environment
```bash
# Create test environment
python -m venv test-env
source test-env/bin/activate
pip install -r requirements-dev.txt

# Run checks
black .
flake8 .
mypy src/
pytest
```