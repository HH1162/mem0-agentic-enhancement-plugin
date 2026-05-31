"""Mem0 Lifecycle Server — bridge layer for access tracking and cleanup.

This module provides a drop-in replacement for the Mem0 SDK that adds:
- Access frequency tracking on search operations
- Exponential decay scoring
- Automated cleanup of stale memories
- Diagnostic commands (stats, least_used)

Usage:
    python -m mem0_lifecycle.server <action> [args...]

Actions:
    search <query> <user_id> [top_k] [rerank]    # Search memories (auto-tracks access)
    add <json_messages> <user_id> <agent_id>     # Add new memory  
    get_all <user_id>                            # Get all memories
    stats [user_id]                              # Show access statistics
    least_used [user_id] [top_n]                 # Show least-used memories
    cleanup [--dry-run] [--threshold X] [user_id] # Remove stale memories below threshold
"""

import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from mem0 import Memory
from qdrant_client import QdrantClient

from .decay import (
    compute_weighted_score,
    should_cleanup,
    HALF_LIFE_DAYS,
    CLEANUP_THRESHOLD,
    ACCESS_COUNT_CAP,
    GRACE_PERIOD_DAYS,
)


class Mem0LifecycleServer:
    """Lifecycle-aware Mem0 client with access tracking and automated cleanup."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with Mem0 configuration.
        
        Args:
            config: Mem0 SDK configuration dict (same format as Memory.from_config)
        """
        self.config = config
        self.memory_client = Memory.from_config(config)
        
        # Extract vector store config for Qdrant connection
        vs_config = config.get('vector_store', {}).get('config', {})
        self.qdrant_client = QdrantClient(
            host=vs_config.get('host', 'localhost'),
            port=vs_config.get('port', 6333)
        )
    
    def search(self, query: str, user_id: str, top_k: int = 5, rerank: bool = True) -> List[Dict]:
        """Search memories and auto-track access frequency.
        
        Every search hit increments access_count and updates last_accessed_at
        in Qdrant's vector payload. Caller sees no API difference.
        
        Args:
            query: Search query text
            user_id: User identifier
            top_k: Number of results to return (default: 5)
            rerank: Enable reranking for better relevance (default: True)
            
        Returns:
            List of matching memories with access tracking applied
        """
        results = self.memory_client.search(query=query, user_id=user_id, limit=top_k, rerank=rerank)
        
        # Track access for each returned memory ID
        for result in results:
            memory_id = result.get('memory_id')
            if not memory_id:
                continue
            
            try:
                payload = result.get('payload', {})
                access_count = payload.get('access_count', 0) + 1
                last_accessed = datetime.now(timezone.utc).isoformat()
                
                # Update Qdrant vector payload
                self.qdrant_client.set_payload(
                    collection_name=self.config['vector_store']['config']['collection_name'],
                    points=[memory_id],
                    payload={
                        'access_count': min(access_count, ACCESS_COUNT_CAP),
                        'last_accessed_at': last_accessed,
                    }
                )
                
                # Also update Mem0 metadata layer for consistency
                self.memory_client.update(
                    memory_id=memory_id,
                    user_id=user_id,
                    metadata={
                        'access_count': min(access_count, ACCESS_COUNT_CAP),
                        'last_accessed_at': last_accessed,
                    }
                )
            except Exception as e:
                print(f"Warning: Failed to track access for {memory_id}: {e}", file=sys.stderr)
        
        return results
    
    def get_all(self, user_id: str) -> List[Dict]:
        """Get all memories for a user."""
        return self.memory_client.get_all(user_id=user_id)
    
    def add(self, messages: List[Dict], user_id: str, agent_id: str = "hermes") -> Dict:
        """Add new memory with LLM extraction."""
        return self.memory_client.add(messages=messages, user_id=user_id, agent_id=agent_id)
    
    def stats(self, user_id: str) -> Dict:
        """Show access statistics for memories."""
        all_memories = self.memory_client.get_all(user_id=user_id)
        
        if not all_memories:
            return {"total": 0, "tracked": 0, "max_count": 0, "avg_count": 0}
        
        total = len(all_memories)
        tracked = 0
        max_count = 0
        total_count = 0
        
        for mem in all_memories:
            payload = mem.get('payload', {})
            count = payload.get('access_count', 0)
            if count > 0:
                tracked += 1
                total_count += count
                max_count = max(max_count, count)
        
        avg_count = total_count / tracked if tracked > 0 else 0
        
        return {
            "total": total,
            "tracked": tracked,
            "max_count": max_count,
            "avg_count": round(avg_count, 1)
        }
    
    def least_used(self, user_id: str, top_n: int = 10) -> List[Dict]:
        """Show least-used memories (by weighted score)."""
        all_memories = self.memory_client.get_all(user_id=user_id)
        
        # Calculate scores for each memory
        scored = []
        for mem in all_memories:
            payload = mem.get('payload', {})
            count = payload.get('access_count', 0)
            last_accessed = payload.get('last_accessed_at', 'never')
            
            score = compute_weighted_score(count, last_accessed)
            scored.append((mem, score))
        
        # Sort by score ascending (least used first)
        scored.sort(key=lambda x: x[1])
        
        return [{"memory": mem, "score": score} for mem, score in scored[:top_n]]
    
    def cleanup(self, user_id: str, dry_run: bool = False, threshold: float = CLEANUP_THRESHOLD) -> Dict:
        """Cleanup stale memories based on weighted score.
        
        Args:
            user_id: User identifier
            dry_run: If True, only preview what would be deleted
            threshold: Score threshold for cleanup (default: CLEANUP_THRESHOLD from decay.py)
            
        Returns:
            Dict with cleanup results including candidates and deleted count
        """
        all_memories = self.memory_client.get_all(user_id=user_id)
        
        cleanup_candidates = []
        for mem in all_memories:
            payload = mem.get('payload', {})
            count = payload.get('access_count', 0)
            last_accessed = payload.get('last_accessed_at', 'never')
            created_at = mem.get('created_at', 'never')
            
            if should_cleanup(count, last_accessed, created_at):
                score = compute_weighted_score(count, last_accessed)
                cleanup_candidates.append((mem, score))
        
        if dry_run:
            return {
                "candidates": [{"memory_id": mem.get('memory_id'), "score": score} for mem, score in cleanup_candidates],
                "count": len(cleanup_candidates)
            }
        
        # Actual cleanup
        deleted = 0
        for mem, score in cleanup_candidates:
            memory_id = mem.get('memory_id')
            try:
                # Delete from both layers to prevent orphans
                self.memory_client.delete(user_id=user_id, memory_id=memory_id)
                deleted += 1
            except Exception as e:
                print(f"Warning: Failed to delete {memory_id}: {e}", file=sys.stderr)
        
        return {
            "deleted": deleted,
            "candidates": len(cleanup_candidates)
        }


def get_memory_client():
    """Create Memory client with local config.

    Replace the values below with your own deployment settings.
    See https://docs.mem0.ai/quick-start for full configuration options.
    """
    config = {
        'llm': {
            'provider': 'openai',
            'config': {
                'api_key': 'your-api-key-here',
                'openai_base_url': 'http://localhost:1234/v1',
                'model': 'qwen3'
            }
        },
        'embedder': {
            'provider': 'huggingface',
            'config': {
                'model': '/path/to/embedding-model'
            }
        },
        'vector_store': {
            'provider': 'qdrant',
            'config': {
                'collection_name': 'mem0',
                'embedding_model_dims': 1024,
                'host': 'localhost',
                'port': 6333
            }
        },
        'graph_store': {
            'provider': 'redis',
            'config': {
                "username": "default",
                "password": "your-redis-password",
                "host": "localhost",
                "port": 6379
            }
        }
    }
    
    return Memory.from_config(config)


def get_qdrant_client():
    """Create Qdrant client with local config."""
    return QdrantClient(
        host='localhost',
        port=6333
    )


def search_with_tracking(memory_client, qdrant_client, query: str, user_id: str, top_k: int = 5, rerank: bool = True):
    """Search memories and auto-track access frequency.

    Every search hit increments access_count and updates last_accessed_at
    in Qdrant's vector payload. Caller sees no API difference.
    """
    results = memory_client.search(query=query, user_id=user_id, limit=top_k, rerank=rerank)
    
    # Track access for each returned memory ID
    for result in results:
        memory_id = result.get('memory_id')
        if not memory_id:
            continue
        
        try:
            payload = result.get('payload', {})
            access_count = payload.get('access_count', 0) + 1
            last_accessed = datetime.now(timezone.utc).isoformat()
            
            # Update Qdrant vector payload
            qdrant_client.set_payload(
                collection_name='mem0',
                points=[memory_id],
                payload={
                    'access_count': min(access_count, ACCESS_COUNT_CAP),
                    'last_accessed_at': last_accessed,
                }
            )
            
            # Also update Mem0 metadata layer for consistency
            memory_client.update(
                memory_id=memory_id,
                user_id=user_id,
                metadata={
                    'access_count': min(access_count, ACCESS_COUNT_CAP),
                    'last_accessed_at': last_accessed,
                }
            )
        except Exception as e:
            print(f"Warning: Failed to track access for {memory_id}: {e}", file=sys.stderr)
    
    return results


def get_stats(memory_client, qdrant_client, user_id: Optional[str] = None):
    """Show access statistics for memories."""
    all_memories = memory_client.get_all(user_id=user_id) if user_id else memory_client.get_all()
    
    if not all_memories:
        print("No memories found")
        return
    
    total = len(all_memories)
    tracked = 0
    max_count = 0
    avg_count = 0
    
    for mem in all_memories:
        payload = mem.get('payload', {})
        count = payload.get('access_count', 0)
        if count > 0:
            tracked += 1
            avg_count += count
            max_count = max(max_count, count)
    
    if tracked > 0:
        avg_count /= tracked
    
    print(f"Total memories: {total}")
    print(f"Tracked (access > 0): {tracked}")
    print(f"Max access count: {max_count}")
    print(f"Average access count: {avg_count:.1f}")


def least_used(memory_client, qdrant_client, user_id: Optional[str] = None, top_n: int = 10):
    """Show least-used memories (by weighted score)."""
    all_memories = memory_client.get_all(user_id=user_id) if user_id else memory_client.get_all()
    
    if not all_memories:
        print("No memories found")
        return
    
    # Calculate scores for each memory
    scored = []
    for mem in all_memories:
        payload = mem.get('payload', {})
        count = payload.get('access_count', 0)
        last_accessed = payload.get('last_accessed_at', 'never')
        
        score = compute_weighted_score(count, last_accessed)
        scored.append((mem, score))
    
    # Sort by score ascending (least used first)
    scored.sort(key=lambda x: x[1])
    
    for mem, score in scored[:top_n]:
        print(f"Score: {score:.3f} | ID: {mem.get('memory_id', 'N/A')[:8]}... | Text: {mem.get('text', '')[:60]}...")


def cleanup_memories(memory_client, qdrant_client, user_id: Optional[str] = None, dry_run: bool = False, threshold: float = CLEANUP_THRESHOLD):
    """Cleanup stale memories based on weighted score."""
    all_memories = memory_client.get_all(user_id=user_id) if user_id else memory_client.get_all()
    
    if not all_memories:
        print("No memories found")
        return
    
    cleanup_candidates = []
    for mem in all_memories:
        payload = mem.get('payload', {})
        count = payload.get('access_count', 0)
        last_accessed = payload.get('last_accessed_at', 'never')
        created_at = mem.get('created_at', 'never')
        
        if should_cleanup(count, last_accessed, created_at):
            score = compute_weighted_score(count, last_accessed)
            cleanup_candidates.append((mem, score))
    
    if not cleanup_candidates:
        print("No cleanup candidates found")
        return
    
    print(f"Found {len(cleanup_candidates)} cleanup candidates")
    
    if dry_run:
        print("DRY RUN - no actual deletion:")
        for mem, score in cleanup_candidates:
            print(f"  Would delete: {mem.get('memory_id', 'N/A')[:8]}... (score: {score:.3f})")
        return
    
    # Actual cleanup
    deleted = 0
    for mem, score in cleanup_candidates:
        memory_id = mem.get('memory_id')
        try:
            # Delete from both layers to prevent orphans
            memory_client.delete(user_id=user_id, memory_id=memory_id)
            deleted += 1
        except Exception as e:
            print(f"Warning: Failed to delete {memory_id}: {e}", file=sys.stderr)
    
    print(f"Deleted {deleted} stale memories")


def cli_main():
    """CLI entry point for mem0_server.py"""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    action = sys.argv[1]
    memory_client = get_memory_client()
    qdrant_client = get_qdrant_client()
    
    if action == 'search':
        query = sys.argv[2]
        user_id = sys.argv[3] if len(sys.argv) > 3 else 'hermes-user'
        top_k = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        rerank = sys.argv[5].lower() == 'true' if len(sys.argv) > 5 else True
        results = search_with_tracking(memory_client, qdrant_client, query, user_id, top_k, rerank)
        print(json.dumps(results, indent=2, ensure_ascii=False))
    
    elif action == 'stats':
        user_id = sys.argv[2] if len(sys.argv) > 2 else None
        get_stats(memory_client, qdrant_client, user_id)
    
    elif action == 'least_used':
        user_id = sys.argv[2] if len(sys.argv) > 2 else None
        top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
        least_used(memory_client, qdrant_client, user_id, top_n)
    
    elif action == 'cleanup':
        args = sys.argv[2:]
        dry_run = '--dry-run' in args
        user_id = args[-1] if len(args) > 1 else None
        cleanup_memories(memory_client, qdrant_client, user_id, dry_run)
    
    else:
        print(f"Unknown action: {action}")
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    cli_main()
