# NeuPRE Usage Guide

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Running Experiments](#running-experiments)
4. [Advanced Usage](#advanced-usage)
5. [Configuration](#configuration)
6. [API Reference](#api-reference)

## Installation

### Step 1: Clone or navigate to NeuPRE directory

```bash
cd NeuPRE
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: For GPU acceleration, ensure CUDA is properly installed.

### Step 3: Verify installation

```bash
python -c "import torch; import gpytorch; import z3; print('Installation successful!')"
```

## Quick Start

### Running the Example

The easiest way to get started is to run the included example:

```bash
python example.py
```

This demonstrates NeuPRE on a simple custom protocol with the following format:
```
[Header(2)] [Command(1)] [Length(1)] [Payload(variable)] [Checksum(1)]
```

Expected output:
- Field segmentation results
- Discovered protocol states
- Inferred field types and constraints
- Protocol specification in `example_output/simple_protocol_spec.txt`

### Using the Command-Line Interface

```bash
# Run example
python main.py example

# Run specific experiment
python main.py experiment 1  # State coverage
python main.py experiment 2  # Segmentation accuracy
python main.py experiment 3  # Constraint inference

# Run on custom data
python main.py run -i path/to/messages -o output_dir --config configs/default_config.yaml
```

## Running Experiments

### Experiment 1: State Coverage Efficiency

**Purpose**: Compare how efficiently NeuPRE vs DYNpre discovers protocol states.

```bash
python experiments/experiment1_state_coverage.py
```

**Output**:
- `state_coverage_efficiency.png`: Curve showing states discovered vs messages sent
- `experiment1_report.txt`: Detailed metrics
- `experiment1_metrics.json`: Raw data

**Key Metrics**:
- **Messages to target coverage**: Number of messages needed to discover 80% of states
- **Efficiency**: Unique states / total messages sent

**Expected Result**: NeuPRE reaches target coverage with ~50% fewer messages.

### Experiment 2: Field Boundary Accuracy

**Purpose**: Evaluate field segmentation accuracy on different protocol types.

```bash
python experiments/experiment2_segmentation.py
```

**Protocols Tested**:
1. **Simple**: Standard protocol with clear field boundaries
2. **High-entropy**: Random-looking fields (simulating encryption)
3. **Mixed**: Text and binary fields combined

**Output**:
- `segmentation_simple.png`, `segmentation_high_entropy.png`, `segmentation_mixed.png`
- `experiment2_report.txt`
- `experiment2_metrics.json`

**Key Metrics**:
- **Precision**: Correct boundaries / predicted boundaries
- **Recall**: Correct boundaries / ground truth boundaries
- **F1-Score**: Harmonic mean of precision and recall
- **Perfect Match Rate**: Percentage of perfectly segmented messages

**Expected Result**: NeuPRE achieves higher F1-Score, especially on high-entropy protocols.

### Experiment 3: Complex Constraint Inference

**Purpose**: Test ability to infer logical constraints between fields.

```bash
python experiments/experiment3_constraints.py
```

**Constraints Tested**:
1. Field_A must be a multiple of Field_B
2. Field_C = XOR(Field_A, Field_B)
3. Field_D = length(Field_E)

**Output**:
- `constraint_inference.png`: Comparison of inference accuracy
- `experiment3_report.txt`
- `experiment3_metrics.json`

**Key Metrics**:
- **Correctly Inferred**: Number of constraints correctly discovered
- **Precision/Recall/F1**: Standard classification metrics

**Expected Result**: NeuPRE discovers all 3 constraints; DYNpre struggles due to random mutation.

## Advanced Usage

### Customizing the Pipeline

```python
from neupre import NeuPRE, setup_logging
import logging

# Setup logging
setup_logging(level=logging.DEBUG, logfile='neupre.log')

# Initialize with custom parameters
neupre = NeuPRE(
    # Format Learner
    ib_d_model=256,           # Larger model
    ib_nhead=8,               # More attention heads
    ib_beta=0.05,             # Less compression

    # State Explorer
    dkl_kappa=3.0,            # More exploration

    # Logic Refiner
    confidence_threshold=0.8,  # Stricter verification

    output_dir='./custom_output'
)
```

### Using Individual Modules

#### Module 1: Format Learner Only

```python
from modules.format_learner import InformationBottleneckFormatLearner

learner = InformationBottleneckFormatLearner(d_model=128, beta=0.1)

# Train on messages
learner.train(messages, responses=None, epochs=50)

# Extract boundaries
boundaries = learner.extract_boundaries(message, threshold=0.5)
segments = learner.segment_message(message)

# Save model
learner.save_model('format_learner.pt')
```

#### Module 2: State Explorer Only

```python
from modules.state_explorer import DeepKernelStateExplorer

explorer = DeepKernelStateExplorer(kappa=2.0)

# Define probe callback
def probe(msg):
    return send_to_server(msg)

# Active exploration
stats = explorer.active_exploration(
    base_messages=seed_messages,
    num_iterations=100,
    probe_callback=probe
)

print(f"Discovered {explorer.get_state_coverage()} states")
```

#### Module 3: Logic Refiner Only

```python
from modules.logic_refiner import NeuroSymbolicLogicRefiner, FieldHypothesis, FieldType

refiner = NeuroSymbolicLogicRefiner(confidence_threshold=0.7)

# Add hypothesis
hypothesis = FieldHypothesis(
    field_index=1,
    field_range=(1, 2),
    field_type=FieldType.LENGTH,
    confidence=0.8,
    parameters={'target_field': 2}
)
refiner.add_hypothesis(hypothesis)

# Verify with SMT solver
def verify(msg):
    accepted = validate_message(msg)
    response = get_response(msg)
    return accepted, response

refiner.refine_rules(template_message, verify)

# Get verified rules
verified = refiner.get_verified_rules()
```

### Implementing Server Callbacks

For real protocol analysis, implement these callbacks:

```python
import socket

class ProtocolClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def send_message(self, msg: bytes) -> bytes:
        """Send message and return response"""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((self.host, self.port))
            sock.sendall(msg)
            response = sock.recv(4096)
            return response
        finally:
            sock.close()

    def verify_message(self, msg: bytes) -> tuple:
        """Send message and check if accepted"""
        response = self.send_message(msg)
        accepted = not response.startswith(b'ERROR')
        return accepted, response

# Use with NeuPRE
client = ProtocolClient('localhost', 8080)

results = neupre.run_full_pipeline(
    initial_messages=messages,
    probe_callback=client.send_message,
    verify_callback=client.verify_message
)
```

## Configuration

### Configuration File Structure

Edit `configs/default_config.yaml`:

```yaml
model:
  format_learner:
    d_model: 128        # Transformer dimension
    nhead: 4            # Attention heads
    num_layers: 3       # Transformer layers
    beta: 0.1           # IB trade-off (0.0-1.0)

  state_explorer:
    embedding_dim: 128  # Byte embedding dimension
    hidden_dim: 256     # LSTM/CNN hidden dimension
    feature_dim: 64     # GP feature dimension
    kappa: 2.0          # UCB exploration (1.0-5.0)

  logic_refiner:
    confidence_threshold: 0.7  # Min confidence (0.0-1.0)
    max_counterexamples: 10    # Z3 attempts

training:
  format_epochs: 50
  format_batch_size: 32
  exploration_iterations: 100
  device: 'cuda'  # or 'cpu'

output:
  save_models: true
  save_plots: true
  log_level: 'INFO'
```

### Parameter Tuning Guidelines

**Beta (IB trade-off)**:
- Lower (0.01-0.05): More compression, simpler segmentation
- Higher (0.2-0.5): Less compression, finer-grained fields
- Default (0.1): Balanced

**Kappa (UCB exploration)**:
- Lower (1.0-1.5): More exploitation, faster convergence
- Higher (3.0-5.0): More exploration, better coverage
- Default (2.0): Balanced

**Confidence Threshold**:
- Lower (0.5-0.6): More hypotheses, more false positives
- Higher (0.8-0.9): Fewer hypotheses, more conservative
- Default (0.7): Balanced

## API Reference

### Main Pipeline

```python
class NeuPRE:
    def __init__(self, ...): ...

    def run_full_pipeline(
        self,
        initial_messages: List[bytes],
        initial_responses: Optional[List[bytes]] = None,
        probe_callback: Optional[Callable] = None,
        verify_callback: Optional[Callable] = None,
        **kwargs
    ) -> Dict: ...

    def phase1_format_learning(...) -> List[List[Tuple]]: ...
    def phase2_state_exploration(...) -> Dict: ...
    def phase3_logic_refinement(...) -> List[FieldHypothesis]: ...

    def save_models(self): ...
    def load_models(self): ...
    def export_protocol_specification(self, filename='protocol_spec.txt'): ...
```

### Evaluation

```python
class NeuPREEvaluator:
    def evaluate_segmentation_accuracy(...) -> SegmentationMetrics: ...
    def evaluate_state_coverage(...) -> StateCoverageMetrics: ...
    def evaluate_constraint_inference(...) -> ConstraintInferenceMetrics: ...

    def plot_state_coverage_curve(...): ...
    def plot_segmentation_comparison(...): ...
    def plot_constraint_inference(...): ...

    def generate_report(...): ...
```

## Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
```
Solution: Reduce batch size or model size in config:
  format_batch_size: 16  # instead of 32
  d_model: 64            # instead of 128
```

**Issue**: Z3 solver timeout
```
Solution: Reduce max_counterexamples:
  max_counterexamples: 5  # instead of 10
```

**Issue**: Poor segmentation accuracy
```
Solution: Increase training epochs or adjust beta:
  format_epochs: 100
  beta: 0.05
```

**Issue**: Slow state exploration
```
Solution: Reduce iterations or increase kappa:
  exploration_iterations: 50
  kappa: 3.0
```

## Best Practices

1. **Start Small**: Test on simple protocols first
2. **Use GPU**: Enable CUDA for significant speedup
3. **Monitor Logs**: Use DEBUG level for detailed information
4. **Validate Results**: Always verify against ground truth when available
5. **Save Models**: Checkpoint models to resume training
6. **Iterate**: Adjust hyperparameters based on results

## Next Steps

- Read the [README](README.md) for overview and theory
- Check [experiments/](experiments/) for detailed evaluation code
- Explore [modules/](modules/) for implementation details
- Customize configurations in [configs/](configs/)

## Support

For issues or questions:
- GitHub Issues: [Project Repository]
- Email: your.email@example.com
- Documentation: [Full docs link]
