"""
MEMOS — Integration Tests
Tests for: FastAPI routes, dependency injection, and evaluation dataset integration.
"""
from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, MagicMock, patch

from app.main import app
from app.api.deps import (
    get_memory_repo,
    get_conv_repo,
    get_user_repo,
    get_vector_repo,
    get_graph_repo,
    get_chat_service,
)
from app.models.domain import (
    Memory,
    MemoryCategory,
    MemoryStatus,
    MemoryType,
    MemoryVersion,
    EvolutionAction,
    Conversation,
    Message,
    MessageRole,
)
from app.schemas.api import ChatResponse
from app.evaluation.dataset_loader import DatasetLoader
from app.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    f1_score,
    duplicate_memory_rate,
    update_detection_accuracy,
    EvaluationReport,
)


@pytest.fixture
def mock_memory_repo():
    repo = MagicMock()
    repo.list_active = AsyncMock(return_value=[
        Memory(
            memory_id="mem_123",
            user_id="demo_user",
            content="User prefers Python for backend development",
            category=MemoryCategory.PREFERENCE,
            importance=0.9,
            confidence=0.85,
        )
    ])
    repo.get_by_id = AsyncMock(side_effect=lambda mid, uid: Memory(
        memory_id=mid,
        user_id=uid,
        content="User prefers Python for backend development",
        category=MemoryCategory.PREFERENCE,
        importance=0.9,
        confidence=0.85,
    ) if mid == "mem_123" else None)
    repo.update = AsyncMock(return_value=True)
    repo.get_versions = AsyncMock(return_value=[
        MemoryVersion(
            version_id="ver_1",
            memory_id="mem_123",
            user_id="demo_user",
            version=1,
            content="User likes Python",
            category=MemoryCategory.PREFERENCE,
            importance=0.9,
            confidence=0.85,
            status=MemoryStatus.ACTIVE,
            evolution_action=EvolutionAction.CREATED,
            change_reason="Initial extraction",
        )
    ])
    repo.get_category_counts = AsyncMock(return_value={"preference": 1})
    return repo


@pytest.fixture
def mock_conv_repo():
    repo = MagicMock()
    repo.list_for_user = AsyncMock(return_value=[
        Conversation(
            conversation_id="conv_1",
            user_id="demo_user",
            title="Python discussion",
            message_count=2,
        )
    ])
    repo.get_messages = AsyncMock(return_value=[
        Message(
            message_id="msg_1",
            conversation_id="conv_1",
            user_id="demo_user",
            role=MessageRole.USER,
            content="I love Python",
        ),
        Message(
            message_id="msg_2",
            conversation_id="conv_1",
            user_id="demo_user",
            role=MessageRole.ASSISTANT,
            content="Python is great!",
            memory_ids_used=["mem_123"],
        ),
    ])
    repo.count_conversations = AsyncMock(return_value=1)
    repo.count_messages = AsyncMock(return_value=2)
    return repo


@pytest.fixture
def mock_vector_repo():
    repo = MagicMock()
    repo.upsert = MagicMock()
    repo.search = MagicMock(return_value=[])
    repo.delete = MagicMock()
    return repo


@pytest.fixture
def mock_graph_repo():
    repo = MagicMock()
    repo.extract_and_merge_entities = AsyncMock()
    repo.find_related_entities = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_chat_service():
    svc = MagicMock()
    svc.handle_chat = AsyncMock(return_value=ChatResponse(
        conversation_id="conv_test_123",
        message_id="msg_resp_123",
        response="Hello! I remember you prefer Python.",
        memories_used=[{
            "memory_id": "mem_123",
            "content": "User prefers Python",
            "category": "preference",
            "importance": 0.9,
            "confidence": 0.85,
        }],
        processing_ms=120,
    ))
    return svc


# ─────────────────────────────────────────────────────────────
# Endpoint Tests
# ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_root_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "MEMOS API"
        assert data["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_health_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "services" in data


@pytest.mark.asyncio
async def test_list_memories(mock_memory_repo):
    app.dependency_overrides[get_memory_repo] = lambda: mock_memory_repo
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/memories?user_id=demo_user")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 1
            assert data[0]["memory_id"] == "mem_123"
            assert data[0]["category"] == "preference"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_memory_by_id(mock_memory_repo):
    app.dependency_overrides[get_memory_repo] = lambda: mock_memory_repo
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/memories/mem_123?user_id=demo_user")
            assert resp.status_code == 200
            data = resp.json()
            assert data["memory_id"] == "mem_123"

            # 404 for nonexistent
            resp_404 = await client.get("/api/memories/nonexistent?user_id=demo_user")
            assert resp_404.status_code == 404
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_update_memory(mock_memory_repo, mock_vector_repo, mock_graph_repo):
    app.dependency_overrides[get_memory_repo] = lambda: mock_memory_repo
    app.dependency_overrides[get_vector_repo] = lambda: mock_vector_repo
    app.dependency_overrides[get_graph_repo] = lambda: mock_graph_repo
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.patch(
                "/api/memories/mem_123?user_id=demo_user",
                json={"content": "User prefers FastAPI and Python", "importance": 0.95},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["content"] == "User prefers FastAPI and Python"
            assert data["importance"] == 0.95
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_memory_history(mock_memory_repo):
    app.dependency_overrides[get_memory_repo] = lambda: mock_memory_repo
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/memories/mem_123/history?user_id=demo_user")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 1
            assert data[0]["version"] == 1
            assert data[0]["evolution_action"] == "created"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_chat_endpoint(mock_chat_service):
    app.dependency_overrides[get_chat_service] = lambda: mock_chat_service
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/chat",
                json={
                    "message": "What language do I like?",
                    "user_id": "demo_user",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "Hello! I remember you prefer Python." in data["response"]
            assert data["conversation_id"] == "conv_test_123"
            assert len(data["memories_used"]) == 1
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_list_conversations(mock_conv_repo):
    app.dependency_overrides[get_conv_repo] = lambda: mock_conv_repo
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/conversations?user_id=demo_user")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 1
            assert data[0]["conversation_id"] == "conv_1"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_get_conversation_messages(mock_conv_repo):
    app.dependency_overrides[get_conv_repo] = lambda: mock_conv_repo
    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/api/conversations/conv_1/messages?user_id=demo_user")
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 2
            assert data[0]["role"] == "user"
            assert data[1]["role"] == "assistant"
    finally:
        app.dependency_overrides.clear()


# ─────────────────────────────────────────────────────────────
# Evaluation Integration & Benchmark Tests
# ─────────────────────────────────────────────────────────────

def test_dataset_loader_and_benchmark():
    loader = DatasetLoader(base_path="..")
    records = loader.load_json("datasets/samples/memos_eval_dataset.json")
    assert len(records) >= 10, f"Expected at least 10 sample scenarios, found {len(records)}"

    # Check each scenario structure
    for record in records:
        assert record.record_id, "Record ID missing"
        assert len(record.conversation) >= 0
        assert len(record.eval_questions) > 0 or len(record.conversation) > 0

    # Test EvaluationReport with simulated results
    report = EvaluationReport(name="Test Benchmark")
    
    r1 = recall_at_k(["m1", "m2", "m3"], {"m1", "m3"}, k=3)
    p1 = precision_at_k(["m1", "m2", "m3"], {"m1", "m3"}, k=3)
    mrr1 = mean_reciprocal_rank(["m1", "m2", "m3"], {"m1", "m3"})
    ndcg1 = ndcg_at_k(["m1", "m2", "m3"], {"m1": 1.0, "m3": 0.8}, k=3)
    
    report.add_result("query_1", "hybrid", {
        "recall@3": r1,
        "precision@3": p1,
        "mrr": mrr1,
        "ndcg@3": ndcg1,
    })

    summary = report.summary()
    assert summary["total_cases"] == 1
    assert "recall@3" in summary["aggregate_metrics"]
    assert summary["aggregate_metrics"]["recall@3"]["mean"] == 1.0
    assert summary["aggregate_metrics"]["mrr"]["mean"] == 1.0

