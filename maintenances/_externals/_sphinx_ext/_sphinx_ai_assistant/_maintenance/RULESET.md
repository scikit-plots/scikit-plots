# `_sphinx_ai_assistant` Durable Ruleset

## Authority and prompt trust

1. Browser code is never a security authority.
2. Documentation/page/retrieval content is untrusted reference data, not system
   instructions.
3. Authoritative model system policy is server-owned and immutable from normal
   client requests.
4. Client-side prompt-injection filters are defense in depth only; direct API
   callers must still be safe.

## Secrets and configuration

5. Production secrets must not be serialized into generated HTML, browser
   globals, URLs, logs, exports, or persistent browser storage.
6. Client config may advertise **capability/presence**, not secret value.
7. Escape-hatch self-hosting behavior must be explicitly labeled insecure/non-
   recommended and may not silently become the default.
8. Every configurable setting has one canonical schema/validation/ownership
   record before settings consolidation is declared complete.

## Service security

9. Credentials are bound to approved destinations; user-controlled routing may
   not receive server credentials.
10. CORS origins are explicit and least-privilege by default.
11. Share read capability and edit/delete capability are distinct.
12. Forwarded client identity is trusted only from an explicitly configured
    proxy boundary.
13. Body/resource limits apply before or while buffering enough to protect
    memory, not only after full allocation.
14. Feedback/training persistence records consent/provenance/authenticity and
    treats user-submitted data as untrusted.
15. Proxy, edge-worker, and direct-model paths must not create an easier bypass
    around each other's policy.

## Representation ownership

16. The assistant owns canonical page Markdown and `llms.txt`, generated at
    `build-finished` over the final HTML.
17. Canonical means the build-time artifact and nothing else. A browser
    conversion is convenience, however faithful, because no external agent can
    fetch it.
18. `VIEW` and `ASK AI` use the canonical static `.md`. `COPY` is convenience by
    default and canonical when its mode toggle selects `static`.
19. The selected representation is observable at the control that produced it;
    convenience output is never labelled canonical.
20. `_sphinx_llm` is frozen and imported by nothing. No assistant surface may
    degrade when it is absent from `extensions`.

## Multi-runtime maintenance

21. Browser JS, worker JS, Python services, Sphinx integration, and schemas all
    receive explicit gates.
22. Known large files are baseline debt; new monolith growth/new responsibility
    triggers review.
23. `_static/_backup` is historical only and may never become a runtime import/
    dependency source.
24. `_maintenance` prototypes are not production behavior until explicitly
    promoted with tests.

## Maintenance truth

25. Desired invariants are not marked `HOLDS` without current-source proof.
26. `ENVIRONMENT_BLOCKED` is not `GREEN`.
27. A security finding is not closed by client-only mitigation if the direct
    service endpoint still violates the invariant.
28. Every closed finding names a regression gate and exact source owner.
