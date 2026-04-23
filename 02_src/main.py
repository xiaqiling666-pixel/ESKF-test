import sys

from eskf_stack.app import main


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
