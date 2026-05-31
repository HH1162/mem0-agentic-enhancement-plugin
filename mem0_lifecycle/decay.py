"""Exponential decay scoring for memory lifecycle management.

Core formula (inspired by Ebbinghaus forgetting curve):

    weighted_score = min(access_count, ACCESS_COUNT_CAP) * 0.5^(days_since_last_access / HALF_LIFE_DAYS)

This mirrors how human memory consolidation works:
- Frequently accessed memories persist longer (like reinforced neural pathways)
- Unused memories naturally decay (like pruned rarely-used connections)
- New memories get protection time (like working memory consolidation)
- Decay is gradual, not abrupt (unlike hard TTL which kills memories instantly)
"""

from datetime import datetime, timezone

# Tunable parameters
HALF_LIFE_DAYS = 7.0       # Exponential decay half-life (days)
CLEANUP_THRESHOLD = 0.05   # Memories with weighted_score < this are candidates for deletion
ACCESS_COUNT_CAP = 255     # Hard cap on access_count — prevents unbounded growth
GRACE_PERIOD_DAYS = 14     # New memories protected from deletion


def compute_weighted_score(access_count: int, last_accessed_iso: str) -> float:
    """Compute exponential decay score for a memory entry.

    Prevents infinite inflation: access_count is capped at 255, so even a
    memory searched thousands of times has a bounded score. Combined with
    half-life decay, old memories naturally fade to zero.

    Timestamp parsing uses Python native fromisoformat + tzinfo check for
    robust handling of UTC, timezone-aware, naive, Z-suffix, and negative-offset
    ISO strings. On parse failure, returns 0.0 (expired) to avoid immortal zombies.

    Examples (half_life=7 days):
    - 3 accesses today -> 3.0
    - 10 accesses, last seen 21 days ago -> 1.25
    - 1 access, last seen 33 days ago -> ~0.05 (cleanup threshold)
    """
    if not access_count or not last_accessed_iso or last_accessed_iso == 'never':
        return 0.0
    
    try:
        ts = last_accessed_iso.replace('Z', '+00:00')
        last_dt = datetime.fromisoformat(ts)
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days = max(0, (now - last_dt).total_seconds() / 86400)
        return float(min(access_count, ACCESS_COUNT_CAP)) * (0.5 ** (days / HALF_LIFE_DAYS))
    except Exception:
        # Timestamp parse failure -> treat as expired to avoid immortal zombies
        return 0.0


def is_grace_protected(created_at_iso: str) -> bool:
    """Check if a memory is within the grace period protection window."""
    if not created_at_iso or created_at_iso == 'never':
        return False
    
    try:
        ts = created_at_iso.replace('Z', '+00:00')
        created_dt = datetime.fromisoformat(ts)
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days = max(0, (now - created_dt).total_seconds() / 86400)
        return days < GRACE_PERIOD_DAYS
    except Exception:
        return False  # If we can't parse, don't protect


def should_cleanup(access_count: int, last_accessed_iso: str, created_at_iso: str) -> bool:
    """Determine if a memory should be cleaned up based on decay score and grace period.

    Returns True if the memory should be deleted, False if it should be kept.
    """
    # Always protect memories within grace period
    if is_grace_protected(created_at_iso):
        return False
    
    score = compute_weighted_score(access_count, last_accessed_iso)
    return score < CLEANUP_THRESHOLD
