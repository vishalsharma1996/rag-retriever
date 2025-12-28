from prometheus_client import Histogram, Counter
# End-to-end batch latency (seconds)
RAG_LATENCY = Histogram(
    "rag_pipeline_latency_seconds",
    "End-to-end batch latency",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 240,300,360,420,480,540,600)
)

RAG_REQUESTS = Counter(
    "rag_requests_total",
    "Query counts by execution path",
    labelnames=["path"]
)
