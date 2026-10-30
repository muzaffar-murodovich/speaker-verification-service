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
# The D13 retrieval benchmark keeps its 152,174 clip vectors apart from real profiles.
BENCH_PREFIX = "spk_bench_"
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
# SVS guide §7.4 and §7.7: int8 scalar quantization, kept in RAM, originals kept too.
INT8 = models.ScalarQuantization(
    scalar=models.ScalarQuantizationConfig(
        type=models.ScalarType.INT8, quantile=0.99, always_ram=True
    )
)


class ModelVersionMismatch(ValueError):
    code = "E_SV_MODEL_VERSION_MISMATCH"


def collection_name(model_version: str, prefix: str = PROFILE_PREFIX) -> str:
    """The prefix (spk_profile_) + the version, lowercased, every run of other characters as one _.

    That slug alone would give "ecapa-v3" and "ecapa_v3" one collection, so the first 8
    hex digits of the exact version string's SHA-256 follow it.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", model_version.lower()).strip("_")
    if not slug:
        raise ValueError(f"model version {model_version!r} has no letters or digits")
    digest = hashlib.sha256(model_version.encode()).hexdigest()[:8]
    return f"{prefix}{slug}_{digest}"


def point_id(profile_id: str) -> str:
    """Qdrant ids are integers or UUIDs; derive a fixed UUID from the profile id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, profile_id))


class ProfileStore:
    def __init__(
        self, client: QdrantClient, model_version: str = MODEL_VERSION, prefix: str = PROFILE_PREFIX
    ) -> None:
        self.client = client
        self.model_version = model_version
        self.collection = collection_name(model_version, prefix)

    def _check_version(self, model_version: str) -> None:
        if model_version != self.model_version:
            raise ModelVersionMismatch(
                f"{model_version!r} vector against the {self.model_version!r} collection"
            )

    def create(self, quantized: bool = False) -> None:
        """Create the collection and its payload indexes. Fails if it already exists.

        Profiles are full precision by default: the collection is small enough to keep in
        memory as float32 (SVS guide §7.2). quantized=True adds int8 for the search graph;
        the float32 originals stay, and decisions are made on them (svs.retrieve).
        """
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE),
            hnsw_config=HNSW,
            quantization_config=INT8 if quantized else None,
        )
        for field, schema in PAYLOAD_INDEXES.items():
            self.client.create_payload_index(self.collection, field, schema)

    def upsert(
        self, model_version: str, profile_id: str, vector: NDArray[np.float32], payload: dict
    ) -> None:
        self.upsert_many(model_version, [profile_id], vector[None, :], [payload])

    def upsert_many(
        self,
        model_version: str,
        profile_ids: list[str],
        vectors: NDArray[np.float32],
        payloads: list[dict],
        batch: int = 1024,
    ) -> None:
        self._check_version(model_version)
        if vectors.shape != (len(profile_ids), EMBEDDING_DIM):
            raise ValueError(
                f"expected {len(profile_ids)} {EMBEDDING_DIM}-dim vectors, got {vectors.shape}"
            )
        if len(payloads) != len(profile_ids):
            raise ValueError(f"{len(payloads)} payloads for {len(profile_ids)} profiles")
        for payload in payloads:
            missing = PAYLOAD_INDEXES.keys() - payload.keys()
            if missing:
                raise ValueError(f"payload lacks {sorted(missing)}")
        for start in range(0, len(profile_ids), batch):
            self.client.upsert(
                self.collection,
                [
                    models.PointStruct(
                        id=point_id(pid), vector=vec.tolist(), payload={**pay, "profile_id": pid}
                    )
                    for pid, vec, pay in zip(
                        profile_ids[start : start + batch],
                        vectors[start : start + batch],
                        payloads[start : start + batch],
                        strict=True,
                    )
                ],
            )

    def search(
        self,
        model_version: str,
        vector: NDArray[np.float32],
        limit: int,
        query_filter: models.Filter | None = None,
        params: models.SearchParams | None = None,
    ) -> list[models.ScoredPoint]:
        self._check_version(model_version)
        return self.client.query_points(
            self.collection,
            query=vector.tolist(),
            limit=limit,
            query_filter=query_filter,
            search_params=params,
            with_payload=True,
            with_vectors=False,
        ).points

    def count(self, model_version: str, query_filter: models.Filter | None) -> int:
        """Exact number of profiles the filter keeps: the candidate pool (SVS guide §4.6)."""
        self._check_version(model_version)
        return self.client.count(self.collection, count_filter=query_filter, exact=True).count

    def get(self, model_version: str, profile_id: str) -> tuple[NDArray[np.float32], dict] | None:
        """One profile's stored vector and payload, or None if it is not in the collection."""
        self._check_version(model_version)
        points = self.client.retrieve(
            self.collection, [point_id(profile_id)], with_payload=True, with_vectors=True
        )
        if not points:
            return None
        return np.array(points[0].vector, np.float32), points[0].payload

    def vectors(self, point_ids: list[str]) -> NDArray[np.float32]:
        """The stored float32 vectors of these points, in the order asked."""
        points = self.client.retrieve(self.collection, point_ids, with_vectors=True)
        by_id = {str(p.id): p.vector for p in points}
        missing = [i for i in point_ids if i not in by_id]
        if missing:
            raise KeyError(f"{len(missing)} points not in {self.collection}, e.g. {missing[0]}")
        return np.array([by_id[i] for i in point_ids], np.float32)


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
