import argparse

from ablation.optimizers.registry import get_optimizer_class, list_optimizers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--optimizer", choices=list_optimizers())
    parser.add_argument("--split")
    parser.add_argument("--basin-limit", type=int)
    parser.add_argument("--population", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--starts", type=int)
    parser.add_argument("--optimizer-seeds")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(f"Generic runner executed for {args.optimizer}")


if __name__ == "__main__":
    main()
