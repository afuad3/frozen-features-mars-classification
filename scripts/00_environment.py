"""Phase A: capture hardware + software environment -> results/environment.md."""
import _bootstrap  # noqa: F401

from src.utils.env import write_environment_md
from src.utils.logging_utils import get_logger


def main():
    log = get_logger("environment")
    out = write_environment_md()
    log.info(f"Wrote {out}")


if __name__ == "__main__":
    main()
