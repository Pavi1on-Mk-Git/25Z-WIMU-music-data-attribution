import argparse
import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help="Path to the original jsonl file.")
    parser.add_argument("--output", type=str, help="Path to the output jsonl file.")
    parser.add_argument("--fraction", type=float, help="Part of the dataset to be sampled.")
    parser.add_argument("--seed", type=int, help="Randomness seed.")
    args = parser.parse_args()

    input_data = pd.read_json(Path(args.input), lines=True)
    output_data = input_data.sample(frac=args.fraction, random_state=args.seed)
    output_data.to_json(args.output, orient="records", lines=True)
