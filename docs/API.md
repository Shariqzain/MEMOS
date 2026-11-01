# MEMOS — REST API Reference Documentation

Base URL: `http://localhost:8000`

---

## 1. System Endpoints

### 1.1 Root
- **Method**: `GET /`
- **Description**: Returns basic application and documentation metadata.
- **Response**:
```json
{
  "name": "MEMOS API",
  "version": "0.1.0",
  "description": "Memory Evolution and Management for Organized Semantic Storage",
  "docs": "/docs",
  "health": "/health"
}
```

### 1.2 Health Check
- **Method**: `GET /health`
- **Description**: Verifies operational status of MongoDB, ChromaDB, Neo4j, and LLM provider.
- **Response**:
```json
{
  "status": "healthy",
  "services": {
    "mongodb": "ok",
    "chromadb": "ok",
    "neo4j": "ok",
    "llm": "ok"
  }
}
```

---

## 2. Chat Endpoints

### 2.1 Send Message
- **Method**: `POST /api/chat`
- **Description**: Sends a message, retrieves relevant memories via hybrid RRF, generates a context-aware response, and schedules memory extraction.
- **Request Body**:
```json
{
  "user_id": "demo_user",
  "conversation_id": "optional-uuid",
  "message": "What is my favorite programming language?",
  "retrieval_mode": "hybrid",
  "stream": false
}
```
- **Response**:
```json
{
  "conversation_id": "conv-1234-5678",
  "message_id": "msg-9876-5432",
  "response": "Based on our past conversations, your favorite programming language is Python.",
  "memories_used": [
    {
      "memory_id": "mem-001",
      "content": "User prefers Python for backend development",
      "category": "preference",
      "importance": 0.95,
      "confidence": 0.90,
      "rrf_score": 0.0327
    }
  ],
  "diagnostics": {
    "semantic_candidates": 3,
    "graph_candidates": 2,
    "final_memories": 1,
    "retrieval_mode": "hybrid",
    "timing_ms": 48
  },
  "processing_ms": 112
}
```

### 2.2 List Conversations
- **Method**: `GET /api/conversations?user_id=demo_user&limit=50`
- **Response**:
```json
[
  {
    "conversation_id": "conv-1234-5678",
    "user_id": "demo_user",
    "title": "Programming language preferences",
    "created_at": "2026-10-24T18:00:00Z",
    "updated_at": "2026-10-24T18:05:00Z",
    "message_count": 4
  }
]
```

### 2.3 Get Conversation Messages
- **Method**: `GET /api/conversations/{conversation_id}/messages?user_id=demo_user`
- **Response**:
```json
[
  {
    "message_id": "msg-1",
    "conversation_id": "conv-1234-5678",
    "role": "user",
    "content": "I really prefer Python for backend development.",
    "memory_ids_used": [],
    "created_at": "2026-10-24T18:00:00Z"
  },
  {
    "message_id": "msg-2",
    "conversation_id": "conv-1234-5678",
    "role": "assistant",
    "content": "Noted! Python is a versatile choice.",
    "memory_ids_used": [],
    "created_at": "2026-10-24T18:00:01Z"
  }
]
```

---

## 3. Memory Management Endpoints

### 3.1 List Memories
- **Method**: `GET /api/memories?user_id=demo_user&category=preference&limit=50&skip=0`
- **Response**:
```json
[
  {
    "memory_id": "mem-001",
    "user_id": "demo_user",
    "content": "User prefers Python for backend development",
    "category": "preference",
    "entities": [
      {"name": "Python", "entity_type": "LANGUAGE"}
    ],
    "importance": 0.95,
    "confidence": 0.90,
    "memory_type": "long_term",
    "status": "active",
    "version": 1,
    "created_at": "2026-10-24T18:00:00Z",
    "updated_at": "2026-10-24T18:00:00Z"
  }
]
```

### 3.2 Get Memory Details
- **Method**: `GET /api/memories/{memory_id}?user_id=demo_user`
- **Response**: Same format as item in memory list.

### 3.3 Update Memory
- **Method**: `PATCH /api/memories/{memory_id}?user_id=demo_user`
- **Request Body**:
```json
{
  "content": "User prefers Python and FastAPI for backend development",
  "importance": 0.98,
  "confidence": 0.95
}
```
- **Response**: Updated memory object with incremented version number.

### 3.4 Soft-Delete Memory
- **Method**: `DELETE /api/memories/{memory_id}?user_id=demo_user`
- **Response**:
```json
{
  "message": "Memory deleted",
  "memory_id": "mem-001"
}
```

### 3.5 Memory Version History
- **Method**: `GET /api/memories/{memory_id}/history?user_id=demo_user`
- **Response**:
```json
[
  {
    "version_id": "ver-001",
    "memory_id": "mem-001",
    "version": 1,
    "content": "User prefers Python",
    "category": "preference",
    "evolution_action": "created",
    "change_reason": "Initial extraction",
    "created_at": "2026-10-24T18:00:00Z"
  },
  {
    "version_id": "ver-002",
    "memory_id": "mem-001",
    "version": 2,
    "content": "User prefers Python and FastAPI",
    "category": "preference",
    "evolution_action": "updated",
    "change_reason": "Manual edit via API",
    "created_at": "2026-10-24T18:15:00Z"
  }
]
```

### 3.6 Restore Memory Version
- **Method**: `POST /api/memories/{memory_id}/restore?version=1&user_id=demo_user`
- **Response**: Restored memory object.

---

## 4. Evaluation & Analytics Endpoints

### 4.1 Evaluation Summary
- **Method**: `GET /evaluation/summary?user_id=demo_user`
- **Response**:
```json
{
  "total_memories": 15,
  "active_memories": 12,
  "superseded_memories": 2,
  "archived_memories": 1,
  "avg_importance": 0.82,
  "avg_confidence": 0.88,
  "total_conversations": 6,
  "total_messages": 24,
  "memory_categories": {
    "preference": 5,
    "fact": 4,
    "project": 3
  }
}
```
