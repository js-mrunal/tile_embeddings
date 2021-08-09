import sys
from pathlib import Path


def initialize_environment():
    BASE_DIR = str(Path("./../src/").resolve())
    if BASE_DIR not in sys.path:
        sys.path.append(BASE_DIR)