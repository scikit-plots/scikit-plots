# `_sphinx_ai_assistant` Runtime Flows

## Page-load / representation flow

```text
HTML page loads
    |
    +--> build-time client-safe config
    +--> browser preferences
    +--> server discovery (non-secret capability truth)
    |
    v
resolve current document representation
    |
    +--> Tier 1 canonical static Markdown from _sphinx_llm
    |       |
    |       +--> manifest / alternate-link lookup
    |
    +--> Tier 2 static compatibility Markdown
    |
    +--> Tier 3 live DOM fallback only if needed
    |
    v
reference context package
```

## Chat flow

```text
user question
     |
browser UI
     |
     +--> user input
     +--> selected documentation context (UNTRUSTED REFERENCE)
     |
     v
proxy/service
     |
     +--> authenticate / authorize / limit / route
     +--> inject immutable SERVER SYSTEM POLICY
     +--> keep user/reference roles non-authoritative
     |
     v
model service/provider
     |
     v
validated/streamed response
     |
     v
browser rendering
```

## Share flow

```text
conversation payload
  -> server validation/size policy
  -> create read locator/capability
  -> return read link

edit/delete
  -> separate write authorization/capability
  -> never infer write authority from read UUID alone
```

## Feedback/training flow

```text
user feedback
  -> explicit consent state/version
  -> schema validation + resource limits
  -> provenance/authenticity metadata
  -> persist as UNTRUSTED CONTRIBUTION
  -> dedup/review/promotion policy outside raw submission path
```

## Failure UX rule

A security or representation failure must not silently fall back to a less safe
path. Fallbacks are explicit, fidelity-labeled, and limited to representation
availability; auth/policy failures remain failures.
