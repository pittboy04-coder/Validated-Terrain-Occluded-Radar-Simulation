#!/usr/bin/env python3
"""CLI entry point for batch generation of synthetic radar training data.

Usage:
    python generate_training_data.py -n 100 -o training_data/
"""
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from radar_sim.batch.generator import BatchGenerator
from radar_sim.batch.config import BatchConfig


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic radar PPI training data with annotations."
    )
    parser.add_argument('-n', '--num-scenarios', type=int, default=10,
                        help='Number of scenarios to generate (default: 10)')
    parser.add_argument('-o', '--output-dir', type=str, default='training_data',
                        help='Output directory (default: training_data/)')
    parser.add_argument('--min-vessels', type=int, default=2,
                        help='Minimum vessels per scenario (default: 2)')
    parser.add_argument('--max-vessels', type=int, default=8,
                        help='Maximum vessels per scenario (default: 8)')
    parser.add_argument('--image-size', type=int, default=512,
                        help='PPI image size in pixels (default: 512)')
    parser.add_argument('--terrain-prob', type=float, default=0.3,
                        help='Probability of terrain per scenario (default: 0.3)')
    parser.add_argument('--coastline-prob', type=float, default=0.2,
                        help='Probability of coastline per scenario (default: 0.2)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)

    config = BatchConfig(
        min_vessels=args.min_vessels,
        max_vessels=args.max_vessels,
        terrain_probability=args.terrain_prob,
        coastline_probability=args.coastline_prob,
        image_size=args.image_size,
    )

    generator = BatchGenerator(config)
    print(f"Generating {args.num_scenarios} scenarios to {args.output_dir}/")
    output_files = generator.generate(args.num_scenarios, args.output_dir)
    print(f"Done. Generated {len(output_files)} scenario(s).")


if __name__ == '__main__':
    main()
