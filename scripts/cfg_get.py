"""Print one value from a pipeline config, for shell scripts (so no script hardcodes a setting).

    python scripts/cfg_get.py <config.yaml> <dotted.key> [default]

Resolves `base:` inheritance, `{seq}` and ${ENV} like every other script. Lists print space-separated,
booleans print 1/0, missing keys print the default (or an empty line).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pose_utils as pu  # noqa: E402


def main():
    if len(sys.argv) < 3:
        print(__doc__); return 2
    v = pu.load_config(sys.argv[1])
    for k in sys.argv[2].split("."):
        v = v.get(k) if isinstance(v, dict) else None
        if v is None:
            break
    if v is None:
        v = sys.argv[3] if len(sys.argv) > 3 else ""
    if isinstance(v, bool):
        v = int(v)
    if isinstance(v, list):
        v = " ".join(map(str, v))
    print(v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
