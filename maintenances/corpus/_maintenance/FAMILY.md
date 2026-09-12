# Corpus family boundaries

Corpus is the protocol-neutral evidence/retrieval foundation. It does not own every consumer of those contracts.

```text
Corpus RetrievalResponse/RetrievalStatus -> MCP adapter/wire layer
Corpus VectorIndexBackend                -> Annoy backend adapter/native mechanics
Corpus CapabilityStatus/Report           -> CLI presentation
```

Rules:

- Do not move MCP wire shapes into Corpus to satisfy an adapter.
- Do not move native Annoy index lifetime/mmap mechanics into Corpus identity or generation logic.
- Do not let CLI re-probe or reinterpret a capability truth that Corpus already owns.
- Do not claim any boundary is currently verified from this partial archive; consumer/runtime evidence is not present here.
- A future complete snapshot may refine these edges, but changes require reproduced source evidence rather than inherited prose.
