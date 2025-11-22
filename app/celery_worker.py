from celery import Celery
celery_app = Celery(
    "rag_pipeline",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
    )
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_backend="redis://localhost:6379/0",
    broker_url="redis://localhost:6379/0",
    timezone="UTC",
    enable_utc=True,
)
celery_app.autodiscover_tasks(['app'])
