# cython: language_level=3
"""BFS on adjacency list (Python objects) - learning template."""
from collections import deque

cpdef list bfs(object adj, int start):
    cdef set seen = set([start])
    q = deque([start])
    out = []
    while q:
        v = q.popleft()
        out.append(v)
        for u in adj[v]:
            if u not in seen:
                seen.add(u)
                q.append(u)
    return out
