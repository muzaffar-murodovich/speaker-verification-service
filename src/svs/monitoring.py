"""What GET /metrics exposes (handbook M4), with the names of SVS guide §11.1.

The text is the Prometheus format: a scraper reads it every few seconds and keeps the
history, so this process only counts since it started. Each Metrics has its own registry,
so two apps in one process (the tests) do not add to each other's counts.
"""

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        # The library's default buckets, 5 ms to 10 s: one item is 0.1-2 s on a CPU.
        self.embed_seconds = Histogram(
            "svs_embed_duration_seconds",
            "Time to embed one /v1/embed item",
            ["model_version", "purpose"],
            registry=self.registry,
        )
        # The library adds _total: svs_embed_rejected_total.
        self.embed_rejected = Counter(
            "svs_embed_rejected",
            "/v1/embed items the usability gate refused, by E_SV_* code",
            ["reason"],
            registry=self.registry,
        )
        # 120 ms is a bucket edge: the §11.1 alert is p95 > 120 ms.
        self.identify_seconds = Histogram(
            "svs_identify_duration_seconds",
            "Time of one identify stage: count (the pool), recall (HNSW), exact (rescore)",
            ["stage"],
            buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.12, 0.25, 0.5, 1.0),
            registry=self.registry,
        )
        # Edges at the pool tiers' scale and at CFG_SV_MAX_AUTO_POOL (50,000).
        self.identify_pool = Histogram(
            "svs_identify_pool_size",
            "Profiles the identify filter kept",
            buckets=(0, 1, 10, 100, 1_000, 10_000, 50_000, 100_000, 1_000_000),
            registry=self.registry,
        )
        self.decisions = Counter(
            "svs_decisions",
            "Bands given: verify's band, or identify's top band with its decision",
            ["call", "band", "decision"],
            registry=self.registry,
        )

    def text(self) -> bytes:
        return generate_latest(self.registry)
