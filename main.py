"""
Main entry point for NeuPRE
"""

import argparse
import yaml
import logging
from pathlib import Path

from neupre import NeuPRE, setup_logging


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='NeuPRE: Neuro-Symbolic Protocol Reverse Engineering'
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run NeuPRE pipeline')
    run_parser.add_argument('-c', '--config', default='configs/default_config.yaml',
                           help='Configuration file')
    run_parser.add_argument('-i', '--input', required=True,
                           help='Input messages file or directory')
    run_parser.add_argument('-o', '--output', default='./neupre_output',
                           help='Output directory')
    run_parser.add_argument('--server-host', help='Protocol server host')
    run_parser.add_argument('--server-port', type=int, help='Protocol server port')

    # Experiment command
    exp_parser = subparsers.add_parser('experiment', help='Run experiments')
    exp_parser.add_argument('experiment_id', type=int, choices=[1, 2, 3],
                           help='Experiment ID (1=coverage, 2=segmentation, 3=constraints)')
    exp_parser.add_argument('-o', '--output', default='./experiments/results',
                           help='Output directory')
    exp_parser.add_argument('--real-data', action='store_true',
                           help='Use real pcap data (for experiment 2)')
    exp_parser.add_argument('--num-samples', type=int, default=100,
                           help='Number of samples per protocol')
    exp_parser.add_argument('--use-dynpre-gt', action='store_true',
                           help='Use DynPRE ground truth for fair comparison (for experiment 2)')

    # Example command
    example_parser = subparsers.add_parser('example', help='Run example')

    args = parser.parse_args()

    if args.command == 'example':
        # Run example
        import example
        example.main()

    elif args.command == 'experiment':
        # Run experiment
        if args.experiment_id == 1:
            from experiments import experiment1_state_coverage
            experiment1_state_coverage.run_experiment1(output_dir=args.output)
        elif args.experiment_id == 2:
            from experiments import experiment2_segmentation
            experiment2_segmentation.run_experiment2(
                num_samples=args.num_samples,
                output_dir=args.output,
                use_real_data=args.real_data,
                use_dynpre_ground_truth=args.use_dynpre_gt
            )
        elif args.experiment_id == 3:
            from experiments import experiment3_constraints
            experiment3_constraints.run_experiment3(output_dir=args.output)

    elif args.command == 'run':
        # Load config
        config = load_config(args.config)

        # Setup logging
        log_level = getattr(logging, config['output']['log_level'])
        setup_logging(level=log_level)

        logging.info("Starting NeuPRE with configuration:")
        logging.info(yaml.dump(config, default_flow_style=False))

        # Initialize NeuPRE
        neupre = NeuPRE(
            ib_d_model=config['model']['format_learner']['d_model'],
            ib_nhead=config['model']['format_learner']['nhead'],
            ib_layers=config['model']['format_learner']['num_layers'],
            ib_beta=config['model']['format_learner']['beta'],
            dkl_embedding_dim=config['model']['state_explorer']['embedding_dim'],
            dkl_hidden_dim=config['model']['state_explorer']['hidden_dim'],
            dkl_feature_dim=config['model']['state_explorer']['feature_dim'],
            dkl_kappa=config['model']['state_explorer']['kappa'],
            confidence_threshold=config['model']['logic_refiner']['confidence_threshold'],
            max_counterexamples=config['model']['logic_refiner']['max_counterexamples'],
            device=config['training']['device'],
            output_dir=args.output
        )

        # Load messages
        input_path = Path(args.input)
        messages = []

        if input_path.is_file():
            # Load from single file
            with open(input_path, 'rb') as f:
                data = f.read()
                # Simple parser: messages separated by newline
                messages = [bytes.fromhex(line.strip()) for line in data.decode().split('\n')
                           if line.strip()]
        elif input_path.is_dir():
            # Load from directory
            for file_path in sorted(input_path.glob('*.bin')):
                with open(file_path, 'rb') as f:
                    messages.append(f.read())

        logging.info(f"Loaded {len(messages)} messages")

        # Define callbacks (requires server implementation)
        # This is a placeholder - users need to implement their own
        def probe_callback(msg: bytes) -> bytes:
            # TODO: Implement actual server communication
            logging.warning("Using dummy probe callback")
            return b'RESPONSE'

        def verify_callback(msg: bytes):
            # TODO: Implement actual server communication
            logging.warning("Using dummy verify callback")
            return True, b'RESPONSE'

        # Run pipeline
        results = neupre.run_full_pipeline(
            initial_messages=messages,
            probe_callback=probe_callback if args.server_host else None,
            verify_callback=verify_callback if args.server_host else None,
            format_epochs=config['training']['format_epochs'],
            format_batch_size=config['training']['format_batch_size'],
            exploration_iterations=config['training']['exploration_iterations'],
            exploration_mutations=config['training']['exploration_mutations']
        )

        # Export results
        neupre.export_protocol_specification()

        if config['output']['save_models']:
            neupre.save_models()

        logging.info("NeuPRE completed successfully")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
