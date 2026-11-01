import json
import sys

from opssleuth.core.logging import get_logger
from opssleuth.ingestion.pipeline import ingest_all

logger = get_logger(__name__)


def main() -> None:
    logger.info("Starting ingestion pipeline")
    stats = ingest_all()
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    sys.exit(main())
