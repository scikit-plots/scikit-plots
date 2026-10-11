# Fresh-chat handoff — Run 25 / B44

Run 25 closes provider-control response buffering and strengthens semantic context integrity.

Core invariants:

1. Storage-provider control responses are bounded before JSON parse; status-only mutations never consume unused bodies.
2. Default control-response ceiling is 4 MiB with a 16 MiB hard clamp.
3. Hugging Face Hub control traffic receives a scoped bounded lower-level client; the previous SDK client factory is restored immediately after each call.
4. Only GitLab supports custom `api_base`, and that authority is validated as HTTPS host-only authority with safe path semantics.
5. Live rendered DOM style/geometry is the pre-serialization visibility authority in same-origin and isolated modes.
6. `SEC-P2-48` and `SEC-P2-49` remain explicit residuals; do not overclaim them as closed.

Proxy version: **7.2.0**.

Before delivery, replay dedicated B44, all registered Node, mutation/privacy, runnable non-Sphinx, compile/syntax/TOML/supply-chain/release-evidence/maintenance, and the Sphinx-inclusive boundary from independently extracted prefinal and final ZIP bytes. Then compare final ZIP contents against exact Run 24 and record immutable SHA-256.
