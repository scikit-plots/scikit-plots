# `_sphinx_ai_assistant` Security Model

## Trust hierarchy

```text
SERVER-OWNED SECURITY/MODEL POLICY
        |
        +-- validates/authenticates/rate-limits/routes
        |
        +-- accepts USER INPUT as untrusted
        |
        +-- accepts DOCUMENT/RETRIEVAL CONTEXT as untrusted evidence

BROWSER
        presentation, local preferences, request UX
        never authoritative for secrets/auth/model policy
```

## Critical boundary families

### Secret boundary

Production credentials remain server-side. Client-visible configuration may
expose non-secret endpoint/capability metadata and booleans such as credential
presence, never values.

### Destination boundary

A server-held credential may be sent only to an approved/bound destination.
Configurable URLs must not turn a server token into an exfiltration primitive.

### Prompt boundary

The direct model/service endpoint enforces system-role ownership. A malicious
page, browser client, or API caller must not gain system authority by choosing a
message role or by embedding instructions in documentation.

### Authorization boundary

Possession of a share/read locator is not edit/delete authority. Write actions
need a distinct capability/authorization decision.

### Origin/identity boundary

CORS and forwarded-client identity have explicit trusted boundaries. The worker
and proxy cannot disagree in a way that creates a bypass path.

### Contribution boundary

Feedback/training data is attacker-controlled input until validated and stored
with consent/provenance/authenticity metadata. Persistence does not make it
trusted training evidence.

### Resource boundary

Body, streaming, concurrency, timeout, and storage limits protect allocation and
processing paths early enough to resist exhaustion.

## Static representation interaction

Using the build-time static Markdown reduces live-DOM variability and
makes page artifacts reviewable, but it does not neutralize malicious prose.
Every representation remains untrusted model context.
