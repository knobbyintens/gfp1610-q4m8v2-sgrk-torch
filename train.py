"""Training entry point for the SGRK recurrence."""

import argparse

from sgrk.layer import SGRKLayer
from sgrk.decay import HorizonDecay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", default="16k")
    parser.add_argument("--sparsity", type=int, default=4)
    args = parser.parse_args()

    layer = SGRKLayer(kernel_bank=8, sparsity=args.sparsity)
    decay = HorizonDecay(channels=layer.channels)
    print(f"training SGRK on horizon={args.horizon} with s={args.sparsity}")


if __name__ == "__main__":
    main()
