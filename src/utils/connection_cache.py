"""
GPU-Accelerated Connection Caching for Hamiltonian Operations.

This module provides caching mechanisms to avoid recomputing Hamiltonian
connections for configurations that have been seen before.

Key optimizations:
- Integer encoding for O(1) hash lookups
- LRU-style eviction for memory management
- Batch operations for efficiency
"""

import torch
from typing import Dict, Tuple, Optional, List


class ConnectionCache:
    """
    Cache for Hamiltonian connections using integer encoding.

    Key insight: For a fixed Hamiltonian, connections for a configuration
    never change. Caching them avoids expensive recomputation.

    Uses integer encoding of configurations for fast hash lookups:
    config -> integer: sum(config[i] * 2^(n-1-i))

    Args:
        num_sites: Number of sites/qubits
        max_cache_size: Maximum number of cached entries
        device: Torch device
    """

    def __init__(
        self,
        num_sites: int,
        max_cache_size: int = 100000,
        device: str = 'cuda'
    ):
        self.num_sites = num_sites
        self.max_cache_size = max_cache_size
        self.device = device

        # Powers of 2 for integer encoding (precomputed on CPU for speed)
        self.powers = (2 ** torch.arange(num_sites - 1, -1, -1, dtype=torch.int64)).numpy()

        # Cache storage: key -> (connected_configs, matrix_elements)
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

        # Access counter for LRU eviction
        self._access_count: Dict[int, int] = {}
        self._total_accesses = 0

        # Statistics
        self.hits = 0
        self.misses = 0

    def _encode_config(self, config: torch.Tensor) -> int:
        """Encode a single configuration as integer."""
        config_np = config.cpu().numpy().astype(int)
        return int((config_np * self.powers).sum())

    def _encode_batch(self, configs: torch.Tensor) -> List[int]:
        """Encode batch of configurations as integers."""
        configs_np = configs.cpu().numpy().astype(int)
        return [(row * self.powers).sum() for row in configs_np]

    def get(
        self, config: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get cached connections for a configuration.

        Args:
            config: (num_sites,) configuration tensor

        Returns:
            (connected, elements) if cached, None otherwise
        """
        key = self._encode_config(config)

        if key in self._cache:
            self.hits += 1
            self._total_accesses += 1
            self._access_count[key] = self._total_accesses
            return self._cache[key]

        self.misses += 1
        return None

    def put(
        self,
        config: torch.Tensor,
        connected: torch.Tensor,
        elements: torch.Tensor
    ):
        """
        Cache connections for a configuration.

        Args:
            config: (num_sites,) configuration tensor
            connected: (n_connections, num_sites) connected configurations
            elements: (n_connections,) matrix elements
        """
        # Evict if at capacity
        if len(self._cache) >= self.max_cache_size:
            self._evict()

        key = self._encode_config(config)
        self._total_accesses += 1
        self._access_count[key] = self._total_accesses

        # Store on GPU for fast retrieval
        self._cache[key] = (
            connected.to(self.device) if len(connected) > 0 else connected,
            elements.to(self.device) if len(elements) > 0 else elements
        )

    def get_or_compute(
        self,
        config: torch.Tensor,
        hamiltonian,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get connections from cache or compute and cache them.

        Args:
            config: (num_sites,) configuration tensor
            hamiltonian: Hamiltonian object with get_connections method

        Returns:
            (connected, elements) tuple
        """
        cached = self.get(config)
        if cached is not None:
            return cached

        # Compute and cache
        connected, elements = hamiltonian.get_connections(config)
        self.put(config, connected, elements)
        return connected, elements

    def get_batch(
        self,
        configs: torch.Tensor,
        hamiltonian,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get connections for a batch of configurations, using cache where available.

        This is the main optimization entry point. Returns all connections
        in a format ready for batched NQS evaluation.

        Args:
            configs: (n_configs, num_sites) configurations
            hamiltonian: Hamiltonian object

        Returns:
            all_connected: (total_connections, num_sites) all connected configs
            all_elements: (total_connections,) all matrix elements
            config_indices: (total_connections,) which original config each belongs to
        """
        n_configs = len(configs)
        device = self.device

        all_connected = []
        all_elements = []
        all_indices = []

        # Encode all configs at once for efficiency
        keys = self._encode_batch(configs)

        for i, key in enumerate(keys):
            # Check cache
            if key in self._cache:
                self.hits += 1
                self._total_accesses += 1
                self._access_count[key] = self._total_accesses
                connected, elements = self._cache[key]
            else:
                self.misses += 1
                # Compute connections
                connected, elements = hamiltonian.get_connections(configs[i])
                # Cache result
                self.put(configs[i], connected, elements)

            n_conn = len(connected)
            if n_conn > 0:
                all_connected.append(connected)
                all_elements.append(elements)
                all_indices.append(
                    torch.full((n_conn,), i, dtype=torch.long, device=device)
                )

        if not all_connected:
            return (
                torch.empty(0, self.num_sites, device=device),
                torch.empty(0, device=device),
                torch.empty(0, dtype=torch.long, device=device)
            )

        return (
            torch.cat(all_connected, dim=0),
            torch.cat(all_elements, dim=0),
            torch.cat(all_indices, dim=0)
        )

    def _evict(self):
        """Evict least recently used entries."""
        # Remove bottom 20% by access time
        n_evict = max(1, self.max_cache_size // 5)

        # Sort by access count (oldest first)
        sorted_keys = sorted(self._access_count.keys(),
                           key=lambda k: self._access_count[k])

        for key in sorted_keys[:n_evict]:
            if key in self._cache:
                del self._cache[key]
            if key in self._access_count:
                del self._access_count[key]

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._access_count.clear()
        self.hits = 0
        self.misses = 0
        self._total_accesses = 0

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def __len__(self) -> int:
        return len(self._cache)

    def stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        return {
            'size': len(self._cache),
            'max_size': self.max_cache_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hit_rate,
        }
