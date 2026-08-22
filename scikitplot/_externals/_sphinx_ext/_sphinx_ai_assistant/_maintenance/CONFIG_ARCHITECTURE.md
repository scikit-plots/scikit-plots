# `_sphinx_ai_assistant` Configuration Architecture

## Current and target must be distinguished

The anchored source has three configuration planes:

```text
Sphinx/build-time config
  conf.py -> __init__.py -> generated window globals
                       |
                       v
Browser runtime config
  generated defaults + browser storage + endpoint/settings UI
                       |
                       v
Server runtime config
  environment/secrets -> discovery + routing/model/persistence policy
```

The **target invariant** is that the server is authoritative for secrets,
authorization, routing trust, and model policy. Build/browser layers carry
client-safe defaults, feature/capability metadata, and user preferences only.

Do not write “secrets are server-only” as a current fact until B03's generated-
HTML and browser-storage leakage gates prove it.

## Canonical setting-definition fields

B10 should converge settings toward a schema with at least:

```text
key
human label/documentation
type
default
scope: build | browser | server | shared
persistence: none | memory | session | local | server
sensitivity: public | internal | credential-presence | secret
client_visible: bool
server_authoritative: bool
validation constraints
deprecation/replacement
serialization target(s)
discovery field if any
```

Schema: `schemas/setting-definition.schema.json`.

## Sensitivity rule

```text
sensitivity=secret
  -> client_visible MUST be false
  -> persistence MUST NOT be local/session browser storage
  -> generated HTML/global serialization forbidden

sensitivity=credential-presence
  -> boolean/capability metadata may be client visible
  -> credential value remains server only
```

## Precedence categories

Do not define one universal precedence order for all settings. Resolve by owner:

- **server-authoritative security/routing**: server policy wins;
- **build presentation defaults**: build defaults may be overridden by safe
  browser preference where explicitly allowed;
- **browser user preferences**: browser registry owns local/session persistence;
- **discovered capabilities**: server manifest describes what is actually
  available; browser cannot promote a disabled capability into authority.

## Discovery contract

Discovery is non-secret. It may publish service/version, model names,
capabilities, safe endpoints, limits/timeouts, and boolean credential presence.
It must not publish keys/tokens/passwords/secrets.

Schema: `schemas/discovery-contract.schema.json`.
