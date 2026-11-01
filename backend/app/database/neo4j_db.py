"""
MEMOS — Neo4j Graph Database Adapter
Handles graph construction, relationship mapping, and graph-based retrieval.
Gracefully degrades when Neo4j is unavailable.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_driver = None
_neo4j_available = False


async def init_neo4j() -> bool:
    """Initialize Neo4j driver and verify connectivity. Returns True if available."""
    global _driver, _neo4j_available

    if not settings.neo4j_enabled:
        logger.info("Neo4j disabled via NEO4J_ENABLED=false.")
        _neo4j_available = False
        return False

    try:
        from neo4j import AsyncGraphDatabase
        _driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            connection_timeout=5,
        )
        async with _driver.session(database=settings.neo4j_database) as session:
            await session.run("RETURN 1")
        await _create_constraints()
        _neo4j_available = True
        logger.info("Neo4j connected: %s", settings.neo4j_uri)
        return True
    except Exception as e:
        logger.warning("Neo4j unavailable: %s. Graph features disabled.", e)
        _neo4j_available = False
        if _driver:
            await _driver.close()
            _driver = None
        return False


async def close_neo4j() -> None:
    global _driver, _neo4j_available
    if _driver:
        await _driver.close()
        _driver = None
    _neo4j_available = False


def is_available() -> bool:
    return _neo4j_available


async def _create_constraints() -> None:
    if not _driver:
        return
    constraints = [
        "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (m:Memory) REQUIRE m.memory_id IS UNIQUE",
        "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.entity_type) IS NODE KEY",
    ]
    async with _driver.session(database=settings.neo4j_database) as session:
        for stmt in constraints:
            try:
                await session.run(stmt)
            except Exception as e:
                logger.debug("Constraint skipped (may already exist): %s — %s", stmt[:60], e)


# ─────────────────────────────────────────────────────────────
# Graph Repository
# ─────────────────────────────────────────────────────────────

class GraphRepository:
    """Neo4j graph operations. All methods no-op gracefully if Neo4j is unavailable."""

    async def upsert_memory(
        self,
        memory_id: str,
        user_id: str,
        content: str,
        category: str,
        importance: float,
        status: str,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
    ) -> None:
        if not _neo4j_available or not _driver:
            return
        try:
            async with _driver.session(database=settings.neo4j_database) as session:
                # Merge User and Memory nodes
                await session.run(
                    """
                    MERGE (u:User {user_id: $uid})
                    MERGE (m:Memory {memory_id: $mid})
                    SET m.content = $content,
                        m.category = $category,
                        m.importance = $importance,
                        m.status = $status,
                        m.user_id = $uid
                    MERGE (u)-[:HAS_MEMORY]->(m)
                    """,
                    uid=user_id, mid=memory_id, content=content[:500],
                    category=category, importance=importance, status=status,
                )
                # Merge entities and link to memory
                for ent in entities:
                    await session.run(
                        """
                        MERGE (e:Entity {name: $name, entity_type: $etype})
                        WITH e
                        MATCH (m:Memory {memory_id: $mid})
                        MERGE (m)-[:MENTIONS]->(e)
                        """,
                        name=ent.get("name", ""), etype=ent.get("entity_type", "entity"), mid=memory_id,
                    )
                # Merge explicit relationships between entities
                for rel in relationships:
                    await session.run(
                        """
                        MERGE (s:Entity {name: $subj, entity_type: 'entity'})
                        MERGE (o:Entity {name: $obj, entity_type: 'entity'})
                        MERGE (s)-[r:RELATED_TO {predicate: $pred}]->(o)
                        SET r.memory_id = $mid
                        """,
                        subj=rel.get("subject", ""), obj=rel.get("obj", ""),
                        pred=rel.get("predicate", "related"), mid=memory_id,
                    )
        except Exception as e:
            logger.error("Neo4j upsert_memory failed for %s: %s", memory_id, e)

    async def mark_superseded(self, old_id: str, new_id: str) -> None:
        if not _neo4j_available or not _driver:
            return
        try:
            async with _driver.session(database=settings.neo4j_database) as session:
                await session.run(
                    """
                    MATCH (old:Memory {memory_id: $old_id})
                    SET old.status = 'superseded'
                    WITH old
                    MATCH (new:Memory {memory_id: $new_id})
                    MERGE (new)-[:SUPERSEDES]->(old)
                    """,
                    old_id=old_id, new_id=new_id,
                )
        except Exception as e:
            logger.error("Neo4j mark_superseded failed: %s", e)

    async def delete_memory(self, memory_id: str) -> None:
        if not _neo4j_available or not _driver:
            return
        try:
            async with _driver.session(database=settings.neo4j_database) as session:
                await session.run(
                    "MATCH (m:Memory {memory_id: $mid}) SET m.status = 'deleted'",
                    mid=memory_id,
                )
        except Exception as e:
            logger.error("Neo4j delete_memory failed: %s", e)

    async def retrieve_by_entities(
        self,
        user_id: str,
        entity_names: List[str],
        depth: int = 2,
        limit: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Graph traversal: find memories connected to given entities.
        Returns list of {memory_id, path_length, entities}.
        """
        if not _neo4j_available or not _driver or not entity_names:
            return []
        try:
            async with _driver.session(database=settings.neo4j_database) as session:
                result = await session.run(
                    """
                    MATCH (e:Entity)
                    WHERE e.name IN $entity_names
                    MATCH path = (e)-[*1..$depth]-(m:Memory)
                    WHERE m.user_id = $uid AND m.status = 'active'
                    WITH m, min(length(path)) AS path_len,
                         collect(DISTINCT e.name) AS matched_entities
                    RETURN m.memory_id AS memory_id,
                           path_len,
                           matched_entities
                    ORDER BY path_len ASC, m.importance DESC
                    LIMIT $limit
                    """,
                    entity_names=entity_names, uid=user_id, depth=depth, limit=limit,
                )
                records = await result.data()
                return [
                    {
                        "memory_id": r["memory_id"],
                        "path_length": r["path_len"],
                        "matched_entities": r["matched_entities"],
                    }
                    for r in records
                ]
        except Exception as e:
            logger.error("Neo4j retrieve_by_entities failed: %s", e)
            return []

    async def get_related_memories(
        self, memory_id: str, user_id: str, limit: int = 5
    ) -> List[str]:
        """Find memories related to a given memory via shared entities."""
        if not _neo4j_available or not _driver:
            return []
        try:
            async with _driver.session(database=settings.neo4j_database) as session:
                result = await session.run(
                    """
                    MATCH (m:Memory {memory_id: $mid})-[:MENTIONS]->(e:Entity)
                    <-[:MENTIONS]-(related:Memory)
                    WHERE related.user_id = $uid
                      AND related.status = 'active'
                      AND related.memory_id <> $mid
                    RETURN DISTINCT related.memory_id AS memory_id
                    LIMIT $limit
                    """,
                    mid=memory_id, uid=user_id, limit=limit,
                )
                records = await result.data()
                return [r["memory_id"] for r in records]
        except Exception as e:
            logger.error("Neo4j get_related_memories failed: %s", e)
            return []

    async def check_health(self) -> bool:
        if not _driver:
            return False
        try:
            async with _driver.session(database=settings.neo4j_database) as session:
                await session.run("RETURN 1")
            return True
        except Exception:
            return False
