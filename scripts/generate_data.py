"""Run the Cinelume generator unchanged, with one optional extra scale.

The generator offers three scales (small ≈6k, medium ≈30k, full ≈120k
accounts) and a fixed default seed. ``full`` needs more than 4 GB of resident
memory, which a laptop with a browser open does not always have; ``medium`` is
small for a six-month test window. This wrapper registers intermediate scales
in the generator's own ``ESCALAS`` table at run time and then calls its
``main()``: no file in the source folder is edited, the seed and the period
are the generator's defaults, and the built-in scales still work.

Usage (from anywhere):
    python scripts/generate_data.py --source ../streaming --scale x60 --saida data/cinelume
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

#: Extra scales, documented in docs/DATA_SOURCE.md.
EXTRA_SCALES = {"x60": {"contas": 60_000}, "x90": {"contas": 90_000}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Cinelume source folder (has gerar_dados.py)")
    known, rest = parser.parse_known_args()

    source = Path(known.source).resolve()
    # The generator resolves its output folders relative to the working directory and
    # imports its package from there, so it runs from its own folder.
    os.chdir(source)
    sys.path.insert(0, str(source))

    from gerador import config as generator_config  # noqa: E402  (needs the path above)

    generator_config.ESCALAS.update(EXTRA_SCALES)
    import gerar_dados  # noqa: E402

    sys.argv = ["gerar_dados.py", *rest]
    gerar_dados.main()


if __name__ == "__main__":
    main()
