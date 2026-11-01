import json
import sys

from opssleuth.ingestion.pipeline import ingest_all


def main() -> None:
    stats = ingest_all()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    sys.exit(main())
