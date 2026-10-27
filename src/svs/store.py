"""Profile centroids in Qdrant, one collection per embedding model version.

The model version is part of the collection name (handbook D11, SVS guide §7.2). A store
object is bound to one version: it only ever addresses its own collection, and it refuses
any vector or query that says it comes from another version. So a cross-version
comparison cannot happen, rather than merely being forbidden.
"""

import argparse
import hashlib
import re
import uuid

import numpy as np
from numpy.typing import NDArray
from qdrant_client import QdrantClient, models

from svs.embed import EMBEDDING_DIM, MODEL_VERSION

PROFILE_PREFIX = "spk_profile_"
# Handbook D11: m=32, ef_construct=256. The handbook is silent on full_scan_threshold;
# 20,000 is SVS guide §7.4 (below it a filtered exact scan beats the graph).
HNSW = models.HnswConfigDiff(m=32, ef_construct=256, full_scan_threshold=20_000)
# Handbook D11 payload fields, each indexed so a filter on it stays fast.
PAYLOAD_INDEXES = {
    "status": models.PayloadSchemaType.KEYWORD,
    "source": models.PayloadSchemaType.KEYWORD,
    "last_seen": models.PayloadSchemaType.DATETIME,
    "evidence_seconds": models.PayloadSchemaType.FLOAT,
}


class ModelVersionMismatch(ValueError):
    code = "E_SV_MODEL_VERSION_MISMATCH"


def collection_name(model_version: str) -> str:
    """spk_profile_ + the version, lowercased, every run of other characters as one _.

    That slug alone would give "ecapa-v3" and "ecapa_v3" one collection, so the first 8
    hex digits of the exact version string's SHA-256 follow it.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", model_version.lower()).strip("_")
    if not slug:
        raise ValueError(f"model version {model_version!r} has no letters or digits")
    digest = hashlib.sha256(model_version.encode()).hexdigest()[:8]
    return f"{PROFILE_PREFIX}{slug}_{digest}"


def point_id(profile_id: str) -> str:
    """Qdrant ids are integers or UUIDs; derive a fixed UUID from the profile id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, profile_id))


class ProfileStore:
    def __init__(self, client: QdrantClient, model_version: str = MODEL_VERSION) -> None:
        self.client = client
        self.model_version = model_version
        self.collection = collection_name(model_version)

    def _check_version(self, model_version: str) -> None:
        if model_version != self.model_version:
            raise ModelVersionMismatch(
                f"{model_version!r} vector against the {self.model_version!r} collection"
            )

    def create(self) -> None:
        """Create the collection and its payload indexes. Fails if it already exists."""
        # Full precision, no quantization: the profile collection is small enough to keep
        # in memory as float32 (SVS guide §7.2), and decisions use full-precision vectors.
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
            hnsw_config=HNSW,
        )
        for field, schema in PAYLOAD_INDEXES.items():
            self.client.create_payload_index(self.collection, field, schema)

    def upsert(
        self, model_version: str, profile_id: str, vector: NDArray[np.float32], payload: dict
    ) -> None:
        self._check_version(model_version)
        if vector.shape != (EMBEDDING_DIM,):
            raise ValueError(f"expected a {EMBEDDING_DIM}-dim vector, got {vector.shape}")
        missing = PAYLOAD_INDEXES.keys() - payload.keys()
        if missing:
            raise ValueError(f"payload lacks {sorted(missing)}")
        self.client.upsert(
            self.collection,
            [
                models.PointStruct(
                    id=point_id(profile_id),
                    vector=vector.tolist(),
                    payload={**payload, "profile_id": profile_id},
                )
            ],
        )

    def search(
        self,
        model_version: str,
        vector: NDArray[np.float32],
        limit: int,
        query_filter: models.Filter | None = None,
    ) -> list[models.ScoredPoint]:
        self._check_version(model_version)
        return self.client.query_points(
            self.collection,
            query=vector.tolist(),
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        ).points


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the profile collection in Qdrant")
    parser.add_argument("--url", default="http://localhost:6333")
    args = parser.parse_args()
    store = ProfileStore(QdrantClient(url=args.url))
    # Re-running is safe: an existing collection is left as it is, and its settings printed.
    if not store.client.collection_exists(store.collection):
        store.create()
    info = store.client.get_collection(store.collection)
    print(store.collection)
    print(info.config.hnsw_config)
    print(sorted(info.payload_schema))


if __name__ == "__main__":
    main()
