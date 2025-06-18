"""uri."""

# pylint: disable=import-outside-toplevel
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-name
# pylint: disable=line-too-long

# ruff: noqa: D103

import os as _os
import pathlib as _pathlib
import posixpath as _posixpath
import re as _re
import urllib as _urllib
import uuid as _uuid
from typing import Any

from ..exceptions import ScikitplotException
from ..utils.os import is_windows

_INVALID_DB_URI_MSG = (
    "Please refer to https://mlflow.org/docs/latest/tracking.html#storage for "
    "format specifications."
)

_DBFS_FUSE_PREFIX = "/dbfs/"
_DBFS_HDFS_URI_PREFIX = "dbfs:/"
_uc_volume_URI_PREFIX = "/Volumes/"  # noqa: N816
_uc_model_URI_PREFIX = "/Models/"  # noqa: N816
_UC_DBFS_SYMLINK_PREFIX = "/.fuse-mounts/"
_DATABRICKS_UNITY_CATALOG_SCHEME = "databricks-uc"
_OSS_UNITY_CATALOG_SCHEME = "uc"


def is_local_uri(uri, is_tracking_or_registry_uri=True):  # noqa: PLR0911
    """
    Return true if the specified URI is a local file path (/foo or file:/foo).

    Parameters
    ----------
    uri :
        The URI.
    is_tracking_or_registry_uri :
        Whether or not the specified URI is an MLflow Tracking or
        MLflow Model Registry URI. Examples of other URIs are MLflow artifact URIs,
        filesystem paths, etc.
    """
    if uri == "databricks" and is_tracking_or_registry_uri:
        return False

    if is_windows() and uri.startswith("\\\\"):
        # windows network drive path looks like: "\\<server name>\path\..."
        return False

    parsed_uri = _urllib.parse.urlparse(uri)
    scheme = parsed_uri.scheme
    if scheme == "":
        return True

    is_remote_hostname = parsed_uri.hostname and not (
        parsed_uri.hostname == "."
        or parsed_uri.hostname.startswith("localhost")
        or parsed_uri.hostname.startswith("127.0.0.1")
    )
    if scheme == "file":
        if is_remote_hostname:
            raise ScikitplotException(
                f"{uri} is not a valid remote uri. For remote access "
                "on windows, please consider using a different scheme "
                "such as SMB (e.g. smb://<hostname>/<path>)."
            )
        return True

    if is_remote_hostname:
        return False

    if (  # noqa: SIM103
        is_windows()
        and len(scheme) == 1
        and scheme.lower() == _pathlib.Path(uri).drive.lower()[0]
    ):  # noqa: SIM103
        return True

    return False


def is_file_uri(uri):
    scheme = _urllib.parse.urlparse(uri).scheme
    return scheme == "file"


def is_http_uri(uri):
    scheme = _urllib.parse.urlparse(uri).scheme
    return scheme in {"http", "https"}


def is_databricks_uri(uri):
    """
    Databricks URIs.

    Look like 'databricks' (default profile) or 'databricks://profile'
    or 'databricks://secret_scope:secret_key_prefix'.
    """
    scheme = _urllib.parse.urlparse(uri).scheme
    return scheme == "databricks" or uri == "databricks"


def is_fuse_or_uc_volumes_uri(uri):
    """
    Validate whether a provided URI is directed to a FUSE mount point or a UC volumes mount point.

    Multiple directory paths are collapsed into a single designator for root path validation.
    For example, "////Volumes/" will resolve to "/Volumes/" for validation purposes.
    """
    resolved_uri = _re.sub("/+", "/", uri).lower()
    return any(
        resolved_uri.startswith(x.lower())
        for x in [
            _DBFS_FUSE_PREFIX,
            _DBFS_HDFS_URI_PREFIX,
            _uc_volume_URI_PREFIX,
            _uc_model_URI_PREFIX,
            _UC_DBFS_SYMLINK_PREFIX,
        ]
    )


def _is_uc_volumes_path(path: str) -> bool:
    return _re.match(r"^/[vV]olumes?/", path) is not None


def is_uc_volumes_uri(uri: str) -> bool:
    parsed_uri = _urllib.parse.urlparse(uri)
    return parsed_uri.scheme == "dbfs" and _is_uc_volumes_path(parsed_uri.path)


def is_valid_uc_volumes_uri(uri: str) -> bool:
    parsed_uri = _urllib.parse.urlparse(uri)
    return parsed_uri.scheme == "dbfs" and bool(
        _re.match(r"^/[vV]olumes?/[^/]+/[^/]+/[^/]+/[^/]+", parsed_uri.path)
    )


def is_databricks_unity_catalog_uri(uri):
    scheme = _urllib.parse.urlparse(uri).scheme
    return _DATABRICKS_UNITY_CATALOG_SCHEME in (scheme, uri)  # noqa: PLR1714


def is_oss_unity_catalog_uri(uri):
    scheme = _urllib.parse.urlparse(uri).scheme
    return scheme == "uc"


def construct_db_uri_from_profile(profile):
    if profile:
        return "databricks://" + profile
    return None


# Both scope and key_prefix should not contain special chars for URIs, like '/'
# and ':'.
def validate_db_scope_prefix_info(scope, prefix):
    for c in ["/", ":", " "]:
        if c in scope:
            raise ScikitplotException(
                f"Unsupported Databricks profile name: {scope}. Profile names cannot contain '{c}'."
            )
        if prefix and c in prefix:
            raise ScikitplotException(
                f"Unsupported Databricks profile key prefix: {prefix}."
                f" Key prefixes cannot contain '{c}'."
            )
    if prefix is not None and prefix.strip() == "":
        raise ScikitplotException(
            f"Unsupported Databricks profile key prefix: '{prefix}'. Key prefixes cannot be empty."
        )


def get_db_info_from_uri(uri):
    """
    Get the Databricks profile specified by the tracking URI (if any).

    Otherwise returns None.
    """
    parsed_uri = _urllib.parse.urlparse(uri)
    if parsed_uri.scheme in ("databricks", _DATABRICKS_UNITY_CATALOG_SCHEME):
        # netloc should not be an empty string unless URI is formatted incorrectly.
        if parsed_uri.netloc == "":
            raise ScikitplotException(
                f"URI is formatted incorrectly: no netloc in URI '{uri}'."
                " This may be the case if there is only one slash in the URI."
            )
        profile_tokens = parsed_uri.netloc.split(":")
        parsed_scope = profile_tokens[0]
        if len(profile_tokens) == 1:
            parsed_key_prefix = None
        elif len(profile_tokens) == 2:  # noqa: PLR2004
            parsed_key_prefix = profile_tokens[1]
        else:
            # parse the content before the first colon as the profile.
            parsed_key_prefix = ":".join(profile_tokens[1:])
        validate_db_scope_prefix_info(parsed_scope, parsed_key_prefix)
        return parsed_scope, parsed_key_prefix
    return None, None


def get_databricks_profile_uri_from_artifact_uri(uri, result_scheme="databricks"):
    """
    Retrieve the netloc portion of the URI as a ``databricks://`` or `databricks-uc://` URI.

    if it is a proper Databricks profile specification, e.g.
    ``profile@databricks`` or ``secret_scope:key_prefix@databricks``.
    """
    parsed = _urllib.parse.urlparse(uri)
    if not parsed.netloc or parsed.hostname != result_scheme:
        return None
    if not parsed.username:  # no profile or scope:key
        return result_scheme  # the default tracking/registry URI
    validate_db_scope_prefix_info(parsed.username, parsed.password)
    key_prefix = ":" + parsed.password if parsed.password else ""
    return f"{result_scheme}://" + parsed.username + key_prefix


def remove_databricks_profile_info_from_artifact_uri(artifact_uri):
    """
    Only removes the netloc portion of the URI.

    if it is a Databricks profile specification, e.g.
    ``profile@databricks`` or ``secret_scope:key_prefix@databricks``.
    """
    parsed = _urllib.parse.urlparse(artifact_uri)
    if not parsed.netloc or parsed.hostname != "databricks":
        return artifact_uri
    return _urllib.parse.urlunparse(parsed._replace(netloc=""))


def add_databricks_profile_info_to_artifact_uri(artifact_uri, databricks_profile_uri):
    """Throws an exception if ``databricks_profile_uri`` is not valid."""
    if not databricks_profile_uri or not is_databricks_uri(databricks_profile_uri):
        return artifact_uri
    artifact_uri_parsed = _urllib.parse.urlparse(artifact_uri)
    # Do not overwrite the authority section if there is already one
    if artifact_uri_parsed.netloc:
        return artifact_uri

    scheme = artifact_uri_parsed.scheme
    if scheme in {"dbfs", "runs", "models"}:
        if databricks_profile_uri == "databricks":
            netloc = "databricks"
        else:
            (profile, key_prefix) = get_db_info_from_uri(databricks_profile_uri)
            prefix = ":" + key_prefix if key_prefix else ""
            netloc = profile + prefix + "@databricks"
        new_parsed = artifact_uri_parsed._replace(netloc=netloc)
        return _urllib.parse.urlunparse(new_parsed)
    return artifact_uri


def extract_and_normalize_path(uri):
    parsed_uri_path = _urllib.parse.urlparse(uri).path
    normalized_path = _posixpath.normpath(parsed_uri_path)
    return normalized_path.lstrip("/")


def append_to_uri_path(uri, *paths):
    """
    Append the specified POSIX `paths` to the path component of the specified `uri`.

    Parameters
    ----------
    uri :
        The input URI, represented as a string.
    paths :
        The POSIX paths to append to the specified `uri`'s path component.

    Returns
    -------
    A new URI with a path component consisting of the specified `paths` appended to
    the path component of the specified `uri`.

    Examples
    --------
    .. code-block:: python
        uri1 = "s3://root/base/path?param=value"
        uri1 = append_to_uri_path(uri1, "some/subpath", "/anotherpath")
        assert uri1 == "s3://root/base/path/some/subpath/anotherpath?param=value"
        uri2 = "a/posix/path"
        uri2 = append_to_uri_path(uri2, "/some", "subpath")
        assert uri2 == "a/posixpath/some/subpath"
    """
    path = ""
    for subpath in paths:
        path = _join_posixpaths_and_append_absolute_suffixes(path, subpath)

    parsed_uri = _urllib.parse.urlparse(uri)

    # Validate query string not to contain any traversal path (../) before appending
    # to the end of the path, otherwise they will be resolved as part of the path.
    validate_query_string(parsed_uri.query)

    if len(parsed_uri.scheme) == 0:
        # If the input URI does not define a scheme, we assume that it is a POSIX path
        # and join it with the specified input paths
        return _join_posixpaths_and_append_absolute_suffixes(uri, path)

    prefix = ""
    if not parsed_uri.path.startswith("/"):
        # For certain URI schemes (e.g., "file:"), urllib's unparse routine does
        # not preserve the relative URI path component properly. In certain cases,
        # urlunparse converts relative paths to absolute paths. We introduce this logic
        # to circumvent urlunparse's erroneous conversion
        prefix = parsed_uri.scheme + ":"
        parsed_uri = parsed_uri._replace(scheme="")

    new_uri_path = _join_posixpaths_and_append_absolute_suffixes(parsed_uri.path, path)
    new_parsed_uri = parsed_uri._replace(path=new_uri_path)
    return prefix + _urllib.parse.urlunparse(new_parsed_uri)


def append_to_uri_query_params(
    uri,
    *query_params: tuple[str, Any],
) -> str:  # noqa: D417
    """
    Append the specified query parameters to an existing URI.

    Parameters
    ----------
    uri :
        The URI to which to append query parameters.
        query_params: Query parameters to append. Each parameter should
        be a 2-element tuple. For example, ``("key", "value")``.
    query_params :
        query_params
    """
    parsed_uri = _urllib.parse.urlparse(uri)
    parsed_query = _urllib.parse.parse_qsl(parsed_uri.query)
    new_parsed_query = parsed_query + list(query_params)
    new_query = _urllib.parse.urlencode(new_parsed_query)
    new_parsed_uri = parsed_uri._replace(query=new_query)
    return _urllib.parse.urlunparse(new_parsed_uri)


def _join_posixpaths_and_append_absolute_suffixes(prefix_path, suffix_path):
    """
    Join the POSIX path `prefix_path` with the POSIX path `suffix_path`.

    Unlike posixpath.join(), if `suffix_path` is an absolute path,
    it is appended to prefix_path.

    Examples
    --------
    >>> result1 = _join_posixpaths_and_append_absolute_suffixes("relpath1", "relpath2")
    >>> assert result1 == "relpath1/relpath2"
    >>> result2 = _join_posixpaths_and_append_absolute_suffixes(
    ...     "relpath", "/absolutepath"
    ... )
    >>> assert result2 == "relpath/absolutepath"
    >>> result3 = _join_posixpaths_and_append_absolute_suffixes(
    ...     "/absolutepath", "relpath"
    ... )
    >>> assert result3 == "/absolutepath/relpath"
    >>> result4 = _join_posixpaths_and_append_absolute_suffixes(
    ...     "/absolutepath1", "/absolutepath2"
    ... )
    >>> assert result4 == "/absolutepath1/absolutepath2"
    """
    if len(prefix_path) == 0:
        return suffix_path

    # If the specified prefix path is non-empty, we must relativize the suffix path by removing
    # the leading slash, if present. Otherwise, posixpath.join() would omit the prefix from the
    # joined path
    suffix_path = suffix_path.lstrip(_posixpath.sep)
    return _posixpath.join(prefix_path, suffix_path)


def is_databricks_acled_artifacts_uri(artifact_uri):
    _ACLED_ARTIFACT_URI = "databricks/mlflow-tracking/"  # noqa: N806
    artifact_uri_path = extract_and_normalize_path(artifact_uri)
    return artifact_uri_path.startswith(_ACLED_ARTIFACT_URI)


def is_databricks_model_registry_artifacts_uri(artifact_uri):
    _MODEL_REGISTRY_ARTIFACT_URI = "databricks/mlflow-registry/"  # noqa: N806
    artifact_uri_path = extract_and_normalize_path(artifact_uri)
    return artifact_uri_path.startswith(_MODEL_REGISTRY_ARTIFACT_URI)


def is_valid_dbfs_uri(uri):
    parsed = _urllib.parse.urlparse(uri)
    if parsed.scheme != "dbfs":
        return False
    try:
        db_profile_uri = get_databricks_profile_uri_from_artifact_uri(uri)
    except ScikitplotException:
        db_profile_uri = None
    return not parsed.netloc or db_profile_uri is not None


def dbfs_hdfs_uri_to_fuse_path(dbfs_uri):
    """
    Convert the provided DBFS URI into a DBFS FUSE path.

    Parameters
    ----------
    dbfs_uri :
        A DBFS URI like "dbfs:/my-directory". Can also be a scheme-less URI like
        "/my-directory" if running in an environment where the default HDFS filesystem
        is "dbfs:/" (e.g. Databricks)

    Returns
    -------
    A DBFS FUSE-style path, e.g. "/dbfs/my-directory"
    """
    if not is_valid_dbfs_uri(dbfs_uri) and dbfs_uri == _posixpath.abspath(dbfs_uri):
        # Convert posixpaths (e.g. "/tmp/mlflow") to DBFS URIs by adding "dbfs:/" as a prefix
        dbfs_uri = "dbfs:" + dbfs_uri
    if not dbfs_uri.startswith(_DBFS_HDFS_URI_PREFIX):
        raise ScikitplotException(
            f"Path '{dbfs_uri}' did not start with expected DBFS URI "
            f"prefix '{_DBFS_HDFS_URI_PREFIX}'",
        )

    return _DBFS_FUSE_PREFIX + dbfs_uri[len(_DBFS_HDFS_URI_PREFIX) :]


def generate_tmp_dfs_path(dfs_tmp):
    return _posixpath.join(dfs_tmp, str(_uuid.uuid4()))


def join_paths(*paths: str) -> str:
    stripped = (p.strip("/") for p in paths)
    return "/" + _posixpath.normpath(_posixpath.join(*stripped))


_OS_ALT_SEPS = [
    sep for sep in [_os.sep, _os.path.altsep] if sep is not None and sep != "/"
]


def _escape_control_characters(text: str) -> str:
    # Method to escape control characters (e.g. \u0017)
    def escape_char(c):
        code_point = ord(c)

        # If it's a control character (ASCII 0-31 or 127), escape it
        if (0 <= code_point <= 31) or (code_point == 127):  # noqa: PLR2004
            return f"%{code_point:02x}"
        return c

    return "".join(escape_char(c) for c in text)


def validate_query_string(query):
    query = _decode(query)
    # Block query strings contain any traversal path (../) because they
    # could be resolved as part of the path and allow path traversal.
    if ".." in query:
        raise ScikitplotException("Invalid query string", error_code=0)


def _decode(url):
    # Keep decoding until the url stops changing (with a max of 10 iterations)
    for _ in range(10):
        decoded = _urllib.parse.unquote(url)
        parsed = _urllib.parse.urlunparse(_urllib.parse.urlparse(decoded))
        if parsed == url:
            return url
        url = parsed

    raise ValueError("Failed to decode url")


def strip_scheme(uri: str) -> str:
    """
    Strip the scheme from the specified URI.

    Examples
    --------
    >>> strip_scheme("http://example.com")
    '//example.com'
    """
    parsed = _urllib.parse.urlparse(uri)
    # `_replace` looks like a private method, but it's actually part of the public API:
    # https://docs.python.org/3/library/collections.html#collections.somenamedtuple._replace
    return _urllib.parse.urlunparse(parsed._replace(scheme=""))


def is_models_uri(uri: str) -> bool:
    try:
        parsed = _urllib.parse.urlparse(uri)
    except ValueError:
        return False

    return parsed.scheme == "models"
