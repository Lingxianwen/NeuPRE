# NeuPRE Project Structure

```
NeuPRE/
│
├── README.md                          # Project overview and quick start
├── USAGE.md                           # Detailed usage guide
├── IMPLEMENTATION.md                  # Technical implementation details
├── PROJECT_SUMMARY.md                 # Project completion report
├── requirements.txt                   # Python dependencies
├── quickstart.sh                      # Automated setup script
├── __init__.py                        # Package initialization
│
├── neupre.py                          # Main NeuPRE pipeline (400 lines)
├── main.py                            # CLI entry point
├── example.py                         # Working example
│
├── modules/                           # Core modules (1,650 lines)
│   ├── __init__.py
│   ├── format_learner.py             # Module 1: Information Bottleneck
│   ├── state_explorer.py             # Module 2: Deep Kernel Learning
│   └── logic_refiner.py              # Module 3: Neuro-Symbolic Logic
│
├── utils/                             # Utilities (450 lines)
│   ├── __init__.py
│   └── evaluator.py                  # Evaluation framework
│
├── experiments/                       # Experiments (850 lines)
│   ├── __init__.py
│   ├── experiment1_state_coverage.py # Exp 1: State coverage efficiency
│   ├── experiment2_segmentation.py   # Exp 2: Field boundary accuracy
│   └── experiment3_constraints.py    # Exp 3: Complex constraint inference
│
└── configs/                           # Configuration (60 lines)
    └── default_config.yaml           # Default configuration

Total: 20 files, ~4,822 lines
```

## File Descriptions

### Documentation Files

- **README.md** (150 lines)
  - Project overview
  - Key features and innovations
  - Comparison with DYNpre
  - Installation instructions
  - Quick start guide
  - Expected results
  - Citation information

- **USAGE.md** (400 lines)
  - Detailed installation steps
  - Running experiments
  - Advanced usage patterns
  - API reference
  - Configuration guide
  - Troubleshooting
  - Best practices

- **IMPLEMENTATION.md** (350 lines)
  - Architecture overview
  - Core algorithms
  - Module descriptions
  - Performance characteristics
  - Comparison table
  - Future extensions

- **PROJECT_SUMMARY.md** (300 lines)
  - Completion report
  - Deliverables checklist
  - Technical achievements
  - File statistics
  - Expected results
  - Research contributions

### Core Implementation Files

- **neupre.py** (~400 lines)
  - `NeuPRE` main class
  - Three-phase pipeline integration
  - `phase1_format_learning()`: Format inference
  - `phase2_state_exploration()`: State discovery
  - `phase3_logic_refinement()`: Constraint verification
  - `run_full_pipeline()`: End-to-end execution
  - Model persistence
  - Protocol specification export

- **modules/format_learner.py** (~600 lines)
  - `TransformerFieldEncoder`: Attention-based encoder
  - `PositionalEncoding`: Positional embeddings
  - `ProtocolDataset`: Data loading
  - `InformationBottleneckFormatLearner`: Main class
  - Information Bottleneck loss computation
  - Boundary extraction from attention
  - Message segmentation
  - Comparison with DYNpre

- **modules/state_explorer.py** (~550 lines)
  - `MessageFeatureExtractor`: LSTM/CNN encoder
  - `DeepKernelGP`: Variational GP model
  - `DeepKernelStateExplorer`: Main class
  - UCB acquisition function
  - Active exploration loop
  - State tracking
  - Bayesian optimization
  - Efficiency comparison

- **modules/logic_refiner.py** (~500 lines)
  - `FieldType` enum: Field type definitions
  - `FieldHypothesis`: Hypothesis data structure
  - `NeuroSymbolicLogicRefiner`: Main class
  - Z3 constraint formulation
  - Counter-example generation
  - Hypothesis verification
  - Field type inference
  - Complex constraint handling

### Utility Files

- **utils/evaluator.py** (~450 lines)
  - `SegmentationMetrics`: Segmentation evaluation
  - `StateCoverageMetrics`: Coverage evaluation
  - `ConstraintInferenceMetrics`: Constraint evaluation
  - `NeuPREEvaluator`: Main evaluator
  - Comparison framework
  - Plotting functions (matplotlib)
  - Report generation (TXT + JSON)

### Experiment Files

- **experiments/experiment1_state_coverage.py** (~250 lines)
  - `MockProtocolServer`: Simulated protocol server
  - `simulate_dynpre_exploration()`: DYNpre baseline
  - `simulate_neupre_exploration()`: NeuPRE method
  - `run_experiment1()`: Main experiment runner
  - Multi-run averaging
  - State coverage curve plotting
  - Metrics computation

- **experiments/experiment2_segmentation.py** (~280 lines)
  - `ProtocolDataset`: Synthetic protocol generators
    - `generate_simple_protocol()`
    - `generate_high_entropy_protocol()`
    - `generate_mixed_protocol()`
  - `simulate_dynpre_segmentation()`: Heuristic-based
  - `simulate_neupre_segmentation()`: IB-based
  - `run_experiment2()`: Main experiment runner
  - Per-protocol evaluation
  - Aggregate statistics

- **experiments/experiment3_constraints.py** (~320 lines)
  - `ComplexProtocolServer`: Protocol with constraints
    - Constraint 1: Multiple relationship
    - Constraint 2: XOR checksum
    - Constraint 3: Length field
  - `simulate_dynpre_constraint_discovery()`: Statistical
  - `simulate_neupre_constraint_discovery()`: Z3 SMT
  - `run_experiment3()`: Main experiment runner
  - Constraint verification
  - Inference accuracy metrics

### Additional Files

- **main.py** (~200 lines)
  - CLI argument parsing
  - Configuration loading (YAML)
  - Subcommands: `run`, `experiment`, `example`
  - Pipeline orchestration
  - Logging setup

- **example.py** (~250 lines)
  - `SimpleProtocolServer`: Example protocol
  - `create_valid_message()`: Message constructor
  - Full pipeline demonstration
  - Results display
  - Self-contained example

- **configs/default_config.yaml** (~60 lines)
  - Model parameters (IB, DKL, Logic)
  - Training configuration
  - Experiment settings
  - Output configuration

- **quickstart.sh** (~50 lines)
  - Virtual environment setup
  - Dependency installation
  - Interactive menu
  - Example execution

## Code Organization Principles

### Modularity
- Each module is self-contained
- Clear interfaces between components
- Minimal coupling

### Extensibility
- Easy to add new field types
- Easy to add new constraint types
- Easy to add new protocols

### Maintainability
- Comprehensive docstrings
- Type hints throughout
- Consistent naming conventions
- Logging at all levels

### Reproducibility
- Fixed random seeds
- Configuration files
- Complete dependency specification
- Automated setup

## Key Statistics

- **Total Lines**: 4,822
- **Python Code**: ~3,400 lines
- **Documentation**: ~1,400 lines
- **Configuration**: ~60 lines
- **Files**: 20
- **Modules**: 3
- **Experiments**: 3
- **Dependencies**: 11

## Execution Flow

```
1. User runs: python main.py experiment 1

2. main.py loads configuration

3. experiment1_state_coverage.py:
   ├── Creates MockProtocolServer
   ├── Simulates DYNpre (random mutation)
   └── Simulates NeuPRE (active learning)
       ├── Uses DeepKernelStateExplorer
       ├── Trains DKL model
       ├── Selects messages via UCB
       └── Discovers states efficiently

4. NeuPREEvaluator:
   ├── Computes metrics
   ├── Generates plots
   └── Creates reports

5. Output in experiments/experiment1_results/
   ├── state_coverage_efficiency.png
   ├── experiment1_report.txt
   └── experiment1_metrics.json
```

## Testing the Implementation

### Quick Test
```bash
cd NeuPRE
python -c "from neupre import NeuPRE; print('Import successful!')"
```

### Run Example
```bash
python example.py
# Expected: Completes in ~1-5 minutes, generates output/
```

### Run Experiments
```bash
python main.py experiment 1  # ~5-10 minutes
python main.py experiment 2  # ~10-15 minutes
python main.py experiment 3  # ~5-10 minutes
```

## Integration with DYNpre

Located in parent directory:
```
.
├── DynPRE-raw/           # DYNpre baseline
│   └── DynPRE/
└── NeuPRE/               # This implementation
    └── (all files above)
```

NeuPRE can compare against DYNpre by:
1. Running DYNpre on same test data
2. Using evaluation framework to compare
3. Generating side-by-side plots

## Dependencies Graph

```
NeuPRE
├── PyTorch (deep learning)
│   └── Transformer, LSTM, CNN
├── GPyTorch (Gaussian Processes)
│   └── Variational GP, DKL
├── Z3 (SMT solver)
│   └── Constraint solving
├── NumPy/SciPy (numerical)
├── Matplotlib (plotting)
├── scikit-learn (metrics)
├── Netzob (protocol analysis)
└── PyYAML (config)
```

## Output Files After Experiments

```
experiments/
├── experiment1_results/
│   ├── state_coverage_efficiency.png
│   ├── experiment1_report.txt
│   └── experiment1_metrics.json
├── experiment2_results/
│   ├── segmentation_simple.png
│   ├── segmentation_high_entropy.png
│   ├── segmentation_mixed.png
│   ├── experiment2_report.txt
│   └── experiment2_metrics.json
└── experiment3_results/
    ├── constraint_inference.png
    ├── experiment3_report.txt
    └── experiment3_metrics.json
```

---

**Last Updated**: 2025-12-12
**Status**: Complete ✅
**Version**: 1.0.0
