"""Profile collection: version in the name, version isolation, index and payload settings."""

import uuid

import numpy as np
import pytest
from qdrant_client import QdrantClient, models

from svs.embed import EMBEDDING_DIM
from svs.store import (
    BENCH_PREFIX,
    PAYLOAD_INDEXES,
    ModelVersionMismatch,
    ProfileStore,
    collection_name,
    point_id,
)

PAYLOAD = {
    "status": "CONFIRMED",
    "source": "cc_north",
    "last_seen": "2026-08-19T09:14:22Z",
    "evidence_seconds": 184.5,
}


def unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(EMBEDDING_DIM).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def client():
    # Local mode: the client keeps points in Python memory. It ignores HNSW settings and
    # payload indexes, so those are tested against the Docker server below.
    return QdrantClient(":memory:")


def test_the_version_is_in_the_collection_name():
    assert collection_name("voxceleb_ECAPA512_LM@a2f3dcb1").startswith(
        "spk_profile_voxceleb_ecapa512_lm_a2f3dcb1_"
    )
    assert collection_name("v1") != collection_name("v2")
    # Same slug, different versions.
    assert collection_name("ecapa-v3") != collection_name("ecapa_v3")
    with pytest.raises(ValueError):
        collection_name("@@")


def test_a_query_from_another_version_is_refused(client):
    store = ProfileStore(client, "v1")
    store.create()
    store.upsert("v1", "spk_a", unit(0), PAYLOAD)
    with pytest.raises(ModelVersionMismatch) as error:
        store.search("v2", unit(0), limit=1)
    assert error.value.code == "E_SV_MODEL_VERSION_MISMATCH"
    with pytest.raises(ModelVersionMismatch):
        store.upsert("v2", "spk_b", unit(1), PAYLOAD)


def test_a_search_never_returns_another_versions_profile(client):
    v1, v2 = ProfileStore(client, "v1"), ProfileStore(client, "v2")
    v1.create()
    v2.create()
    v1.upsert("v1", "spk_a", unit(0), PAYLOAD)
    # The exact same vector under v2: the best possible match, if versions could mix.
    v2.upsert("v2", "spk_b", unit(0), PAYLOAD)
    hits = v1.search("v1", unit(0), limit=10)
    assert [hit.payload["profile_id"] for hit in hits] == ["spk_a"]


def test_upsert_needs_every_payload_field_and_192_dims(client):
    store = ProfileStore(client, "v1")
    store.create()
    with pytest.raises(ValueError, match="evidence_seconds"):
        store.upsert(
            "v1", "spk_a", unit(0), {k: PAYLOAD[k] for k in PAYLOAD if k != "evidence_seconds"}
        )
    with pytest.raises(ValueError, match="192"):
        store.upsert("v1", "spk_a", unit(0)[:100], PAYLOAD)


def test_upsert_twice_replaces_the_profile(client):
    store = ProfileStore(client, "v1")
    store.create()
    store.upsert("v1", "spk_a", unit(0), PAYLOAD)
    store.upsert("v1", "spk_a", unit(1), {**PAYLOAD, "status": "PROVISIONAL"})
    hits = store.search("v1", unit(1), limit=10)
    assert len(hits) == 1
    assert hits[0].payload["status"] == "PROVISIONAL"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_a_filter_on_a_payload_field_drops_the_others(client):
    store = ProfileStore(client, "v1")
    store.create()
    store.upsert("v1", "spk_a", unit(0), PAYLOAD)
    store.upsert("v1", "spk_b", unit(1), {**PAYLOAD, "source": "cc_south"})
    only_south = models.Filter(
        must=[models.FieldCondition(key="source", match=models.MatchValue(value="cc_south"))]
    )
    hits = store.search("v1", unit(0), limit=10, query_filter=only_south)
    assert [hit.payload["profile_id"] for hit in hits] == ["spk_b"]


@pytest.fixture
def server():
    client = QdrantClient(url="http://localhost:6333", timeout=2)
    try:
        client.get_collections()
    except Exception:
        pytest.skip("no Qdrant on localhost:6333 (docker compose up -d)")
    version = f"test-{uuid.uuid4().hex[:8]}"
    yield client, version
    client.delete_collection(collection_name(version))


def test_the_server_collection_has_the_handbook_index_settings(server):
    client, version = server
    store = ProfileStore(client, version)
    store.create()
    info = client.get_collection(store.collection)
    assert info.config.hnsw_config.m == 32
    assert info.config.hnsw_config.ef_construct == 256
    assert info.config.params.vectors.size == EMBEDDING_DIM
    assert info.config.params.vectors.distance == models.Distance.COSINE
    assert {k: v.data_type for k, v in info.payload_schema.items()} == PAYLOAD_INDEXES


def test_vectors_come_back_in_the_order_asked(client):
    store = ProfileStore(client, "v1")
    store.create()
    store.upsert_many("v1", ["spk_a", "spk_b"], np.stack([unit(0), unit(1)]), [PAYLOAD] * 2)
    got = store.vectors([point_id("spk_b"), point_id("spk_a")])
    np.testing.assert_allclose(got, np.stack([unit(1), unit(0)]), atol=1e-6)
    with pytest.raises(KeyError):
        store.vectors([point_id("spk_c")])


def test_upsert_many_refuses_another_version_and_short_payloads(client):
    store = ProfileStore(client, "v1")
    store.create()
    with pytest.raises(ModelVersionMismatch):
        store.upsert_many("v2", ["spk_a"], unit(0)[None, :], [PAYLOAD])
    with pytest.raises(ValueError, match="payloads"):
        store.upsert_many("v1", ["spk_a", "spk_b"], np.stack([unit(0), unit(1)]), [PAYLOAD])


def test_the_benchmark_prefix_is_a_separate_collection(client):
    bench = ProfileStore(client, "v1", BENCH_PREFIX).collection
    assert bench != ProfileStore(client, "v1").collection
    assert bench.startswith(BENCH_PREFIX)


def test_the_quantized_server_collection_is_int8_with_originals(server):
    client, version = server
    store = ProfileStore(client, version)
    store.create(quantized=True)
    info = client.get_collection(store.collection)
    scalar = info.config.quantization_config.scalar
    assert scalar.type == models.ScalarType.INT8
    assert scalar.always_ram
    store.upsert(version, "spk_a", unit(0), PAYLOAD)
    # The original float32 vector is still there for the exact stage.
    np.testing.assert_allclose(store.vectors([point_id("spk_a")])[0], unit(0), atol=1e-6)
