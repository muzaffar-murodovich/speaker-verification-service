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

    def text(self) -> bytes:
        return generate_latest(self.registry)
