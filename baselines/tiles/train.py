"""Binary for invoking the training loop."""

from collections.abc import Sequence

from absl import app

from tpu_graphs.baselines.tiles import train_args
from tpu_graphs.baselines.tiles import train_lib


def main(unused_argv: Sequence[str]) -> None:
  train_lib.train(train_args.get_args())


if __name__ == '__main__':
  app.run(main)
