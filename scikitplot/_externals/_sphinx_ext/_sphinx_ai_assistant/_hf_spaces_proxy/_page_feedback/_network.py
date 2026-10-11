"""
Dependency-free network-address policy shared by the Sphinx adapter and service.

One module decides which feedback endpoints a site may publish and which browser
origins a service may allow, so the two sides cannot drift apart.

Notes
-----
**User notes.** Three policies, from narrow to wide:

``"strict"`` (default)
    HTTPS on port 443, HTTP or HTTPS (any port) on a loopback host, and a
    same-origin path such as ``/v1/feedback``. Right for public sites.
``"private-network"``
    Adds HTTPS on any port and plain HTTP to private-network hosts: RFC 1918,
    CGNAT, link-local and unique-local IP literals, single-label names
    (``feedback``), and names under ``.internal``, ``.local``, ``.localhost``,
    ``.home.arpa`` or an operator-supplied suffix (``.corp.example``). Right for
    intranet, air-gapped and on-premises documentation.
``"any"``
    Adds plain HTTP to any host. Feedback text then crosses networks
    unencrypted; use only where the transport is protected some other way.

A browser blocks an ``http://`` endpoint from an ``https://`` page (mixed
content), whatever the policy says; serve both over the same scheme.

**Developer notes.** Private ranges are listed explicitly rather than taken
from :attr:`ipaddress.IPv4Address.is_private`, whose answer has changed
between Python releases; the decision must be identical on every supported
interpreter. Name matching is purely lexical: no DNS lookup is made, so the
result never depends on the machine that runs the build.
"""

from __future__ import annotations

import ipaddress
import re
from typing import Any, Iterable
from urllib.parse import urlsplit

#: Endpoint/origin policies, narrowest first.
NETWORK_POLICIES = ("strict", "private-network", "any")

#: Name suffixes reserved for local or private use (RFC 6761 ``.localhost``,
#: RFC 6762 ``.local``, RFC 8375 ``.home.arpa``, ICANN-reserved ``.internal``).
RESERVED_PRIVATE_SUFFIXES = (".localhost", ".local", ".internal", ".home.arpa")

_PRIVATE_NETWORKS = tuple(
    ipaddress.ip_network(cidr)
    for cidr in (
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "100.64.0.0/10",  # carrier-grade NAT / overlay VPNs
        "169.254.0.0/16",  # link-local
        "fc00::/7",  # unique-local
        "fe80::/10",  # link-local
    )
)
_LOOPBACK_NETWORKS = tuple(
    ipaddress.ip_network(cidr) for cidr in ("127.0.0.0/8", "::1/128")
)
_LABEL_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_MAX_URL_CHARS = 2048
_MAX_SUFFIXES = 32


class NetworkPolicyError(ValueError):
    """Raised for an address the selected policy does not permit."""


def normalize_policy(value: Any, *, name: str) -> str:
    """
    Return a canonical policy name.

    Parameters
    ----------
    value : str or None
        ``None`` or ``""`` selects ``"strict"``.
    name : str
        Setting name used in error messages.

    Returns
    -------
    str
        One of :data:`NETWORK_POLICIES`.

    Raises
    ------
    NetworkPolicyError
        If *value* is not a string naming a known policy.
    """
    if value in (None, ""):
        return "strict"
    if not isinstance(value, str):
        raise NetworkPolicyError(f"{name} must be a string")
    text = value.strip().lower()
    if text not in NETWORK_POLICIES:
        raise NetworkPolicyError(f"{name} must be one of {list(NETWORK_POLICIES)}")
    return text


def normalize_private_suffixes(value: Any, *, name: str) -> tuple[str, ...]:
    """
    Return operator-supplied private name suffixes in canonical form.

    Parameters
    ----------
    value : str, list of str, tuple of str or None
        Comma-separated string or sequence, for example ``".corp.example"``.
        A missing leading dot is added.
    name : str
        Setting name used in error messages.

    Returns
    -------
    tuple of str
        Lowercase suffixes, each starting with ``.``, without duplicates.

    Raises
    ------
    NetworkPolicyError
        If an entry is not a dot-separated sequence of DNS labels, or there are
        more than 32 entries.
    """
    if value in (None, "", (), []):
        return ()
    if isinstance(value, str):
        items: Iterable[Any] = value.split(",")
    elif isinstance(value, (list, tuple)):
        items = value
    else:
        raise NetworkPolicyError(f"{name} must be a string or a list of strings")
    result: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise NetworkPolicyError(f"{name} entries must be strings")
        text = item.strip().lower()
        if not text:
            continue
        if not text.startswith("."):
            text = "." + text
        labels = text[1:].split(".")
        if not all(_LABEL_RE.fullmatch(label) for label in labels):
            raise NetworkPolicyError(f"{name} contains an invalid suffix: {item!r}")
        if text not in result:
            result.append(text)
    if len(result) > _MAX_SUFFIXES:
        raise NetworkPolicyError(f"{name} contains more than {_MAX_SUFFIXES} suffixes")
    return tuple(result)


def _ip(host: str):
    try:
        return ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return None


def is_loopback_host(host: str) -> bool:
    """Return whether *host* is ``localhost``, ``*.localhost`` or a loopback IP."""
    text = (host or "").lower().rstrip(".")
    address = _ip(text)
    if address is not None:
        return any(address in network for network in _LOOPBACK_NETWORKS)
    return text == "localhost" or text.endswith(".localhost")


def is_private_host(host: str, extra_suffixes: Iterable[str] = ()) -> bool:
    """
    Return whether *host* names a loopback or private-network destination.

    Parameters
    ----------
    host : str
        Host as returned by :func:`urllib.parse.urlsplit` ``hostname``.
    extra_suffixes : iterable of str
        Canonical suffixes from :func:`normalize_private_suffixes`.

    Returns
    -------
    bool
        ``True`` for loopback and private IP literals, single-label names, and
        names under a reserved or extra suffix. No DNS lookup is made.
    """
    text = (host or "").lower().rstrip(".")
    if not text:
        return False
    if is_loopback_host(text):
        return True
    address = _ip(text)
    if address is not None:
        return any(address in network for network in _PRIVATE_NETWORKS)
    if "." not in text:
        return True
    return text.endswith(tuple(RESERVED_PRIVATE_SUFFIXES) + tuple(extra_suffixes))


def _check_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise NetworkPolicyError(f"{name} must be a string")
    text = value.strip()
    if len(text) > _MAX_URL_CHARS or any(
        ord(ch) < 33 or ord(ch) == 127  # ruff: ignore[magic-value-comparison]
        for ch in text
    ):
        raise NetworkPolicyError(f"{name} contains whitespace or control characters")
    return text


def _same_origin_path(text: str, *, name: str) -> str:
    if text.startswith("//") or any(marker in text for marker in ("\\", "?", "#")):
        raise NetworkPolicyError(
            f"{name} same-origin path must be a plain absolute path such as "
            "'/v1/feedback'"
        )
    if any(part in {".", ".."} for part in text.split("/")[1:]):
        raise NetworkPolicyError(f"{name} same-origin path must not contain . or ..")
    return text.rstrip("/") or "/"


def _scheme_allowed(
    scheme: str, host: str, port: int | None, *, policy: str, suffixes: Iterable[str]
) -> bool:
    if is_loopback_host(host):
        return scheme in {"http", "https"}
    if scheme == "https":
        return port in (None, 443) or policy != "strict"
    if scheme == "http":
        if policy == "any":
            return True
        return policy == "private-network" and is_private_host(host, suffixes)
    return False


def validate_endpoint_url(
    value: Any,
    *,
    policy: str = "strict",
    private_suffixes: Iterable[str] = (),
    name: str = "feedback_endpoint",
) -> str:
    """
    Validate a browser-visible feedback endpoint under *policy*.

    Parameters
    ----------
    value : str
        ``""`` (no endpoint), a same-origin path starting with one ``/``, or an
        absolute ``http``/``https`` URL.
    policy : {"strict", "private-network", "any"}, default "strict"
        See the module notes.
    private_suffixes : iterable of str, default ()
        Extra private name suffixes for ``"private-network"``.
    name : str, default "feedback_endpoint"
        Setting name used in error messages.

    Returns
    -------
    str
        The endpoint without a trailing slash, or ``""``.

    Raises
    ------
    NetworkPolicyError
        For credentials, a query or fragment, an unknown scheme, a missing
        host, an invalid port, or a scheme/host/port the policy does not allow.

    Examples
    --------
    >>> validate_endpoint_url("/v1/feedback")
    '/v1/feedback'
    >>> validate_endpoint_url(
    ...     "http://feedback.internal:8080/v1/feedback", policy="private-network"
    ... )
    'http://feedback.internal:8080/v1/feedback'
    """
    text = _check_text(value, name=name)
    if not text:
        return ""
    if text.startswith("/"):
        return _same_origin_path(text, name=name)
    try:
        parsed = urlsplit(text)
        port = parsed.port
    except ValueError as exc:
        raise NetworkPolicyError(f"{name} is not a valid URL") from exc
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"}:
        raise NetworkPolicyError(f"{name} must be an http(s) URL or a same-origin path")
    if parsed.username or parsed.password or parsed.fragment or parsed.query:
        raise NetworkPolicyError(
            f"{name} must not contain credentials, a query, or a fragment"
        )
    if not host:
        raise NetworkPolicyError(f"{name} must include a host")
    if not _scheme_allowed(
        parsed.scheme, host, port, policy=policy, suffixes=private_suffixes
    ):
        raise NetworkPolicyError(_refusal(name, parsed.scheme, policy))
    return text.rstrip("/")


def _refusal(name: str, scheme: str, policy: str) -> str:
    if policy == "strict":
        return (
            f"{name} must use HTTPS on the standard port (localhost HTTP is "
            "allowed); set the policy to 'private-network' for intranet hosts"
        )
    if policy == "private-network":
        return (
            f"{name} uses {scheme} to a host that is not on a private network; "
            "use HTTPS, add the host's suffix to the private suffixes, or set "
            "the policy to 'any'"
        )
    return f"{name} is not allowed"


def normalize_origin(
    value: Any,
    *,
    policy: str = "strict",
    private_suffixes: Iterable[str] = (),
    name: str = "feedback allowed origin",
) -> str:
    """
    Return a canonical browser origin (``scheme://host[:port]``).

    Parameters
    ----------
    value : str
        An origin; a trailing ``/`` is accepted, any other path is not.
    policy, private_suffixes, name
        As for :func:`validate_endpoint_url`.

    Returns
    -------
    str
        Lowercase host, default ports (443/80) dropped, IPv6 bracketed.

    Raises
    ------
    NetworkPolicyError
        For anything that is not a bare origin, or that *policy* forbids.
    """
    text = _check_text(value, name=name)
    if not text:
        raise NetworkPolicyError(f"{name} is invalid")
    try:
        parsed = urlsplit(text)
        port = parsed.port
    except ValueError as exc:
        raise NetworkPolicyError(f"{name} is invalid") from exc
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme not in {"http", "https"}
        or not host
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise NetworkPolicyError(f"{name} must contain only scheme and authority")
    if not _scheme_allowed(
        parsed.scheme, host, port, policy=policy, suffixes=private_suffixes
    ):
        raise NetworkPolicyError(_refusal(name, parsed.scheme, policy))
    authority = f"[{host}]" if ":" in host else host
    default_port = 443 if parsed.scheme == "https" else 80
    if port is not None and port != default_port:
        authority += f":{port}"
    return f"{parsed.scheme}://{authority}"


__all__ = [
    "NETWORK_POLICIES",
    "RESERVED_PRIVATE_SUFFIXES",
    "NetworkPolicyError",
    "is_loopback_host",
    "is_private_host",
    "normalize_origin",
    "normalize_policy",
    "normalize_private_suffixes",
    "validate_endpoint_url",
]
