"""Convenience wrapper for running ingestion without `pip install -e`.

Once the package is installed, prefer the `opssleuth-ingest` console script.
"""

from opssleuth.cli.ingest import main

if __name__ == "__main__":
    main()
