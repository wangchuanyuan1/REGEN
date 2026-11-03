"""命令行入口：训练和推理。"""

from __future__ import annotations

import argparse

from .data import load_dataset
from .pipeline import infer, train


def main() -> None:
    parser = argparse.ArgumentParser(description="REGEN dynamic GRN reconstruction")
    subcommands = parser.add_subparsers(dest="command", required=True)
    train_parser = subcommands.add_parser("train", help="使用带标签数据训练模型")
    infer_parser = subcommands.add_parser("infer", help="使用已训练模型输出 GRN 概率")
    for command in (train_parser, infer_parser):
        command.add_argument("--input", required=True, help="包含 expression.csv 与 time.csv 的目录")
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--device", default="cpu", help="例如 cpu 或 cuda")
    train_parser.add_argument("--windows", type=int, default=4)
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--seed", type=int, default=0)
    infer_parser.add_argument("--output", required=True, help="预测 CSV 的输出路径")
    infer_parser.add_argument("--sampling-steps", type=int, default=40)
    args = parser.parse_args()
    if args.command == "train":
        train(load_dataset(args.input, require_labels=True), args.checkpoint, args.windows, args.epochs, args.device, args.seed)
    else:
        infer(load_dataset(args.input), args.checkpoint, args.output, args.device, args.sampling_steps)


if __name__ == "__main__":
    main()
