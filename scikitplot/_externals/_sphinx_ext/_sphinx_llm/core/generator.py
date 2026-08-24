# SPDX-License-Identifier: BSD-3-Clause
"""Downstream canonical artifact generator built around the preserved NVIDIA core."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import docutils.nodes
from sphinx.errors import ExtensionError
from sphinx.util import logging

from ..adapters.inventory import SemanticNodeInventory
from ..adapters.sphinx import INVENTORY_FILENAME
from ..compat.html_fallback import convert_html_file
from ..compat.markdown_generator import ConfigParityMarkdownGenerator
from ..curation.policy import (
    SizeLimits,
    evaluate_size_policy,
    infer_section,
    order_docnames,
    size_note,
    validate_text_max_bytes,
)
from ..sphinx_llm.version import __version__
from .artifacts import (
    atomic_write_json,
    make_manifest,
    make_provenance,
    normalize_relative_path,
    output_hashes,
    sha256_file,
)
from .augmentation import AugmentationSettings
from .index import IndexPage, render_llms_index

logger = logging.getLogger(__name__)


class CanonicalArtifactGenerator(ConfigParityMarkdownGenerator):
    """Generate Tier-1/Tier-2 artifacts without modifying vendored sources."""

    def __init__(self, app):
        super().__init__(app)
        self._semantic_inventory: SemanticNodeInventory | None = None
        self._description_by_docname: dict[str, tuple[str, str]] = {}
        self._full_included_docnames: set[str] = set()
        self._full_action = "not-run"
        self._full_measure: dict[str, int] = {}
        self._fallback_documents: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # A03/A04: retain child resolved-node inventory before upstream cleanup.
    # ------------------------------------------------------------------
    def copy_markdown_files(self):
        super().copy_markdown_files()
        inventory_path = self.md_build_dir / INVENTORY_FILENAME
        try:
            payload = json.loads(inventory_path.read_text(encoding="utf-8"))
            self._semantic_inventory = SemanticNodeInventory.from_dict(payload)
        except (OSError, TypeError, ValueError):
            logger.error(
                "Canonical Markdown child build produced no readable semantic node inventory"
            )
            self._semantic_inventory = None

    # ------------------------------------------------------------------
    # A07: complete-corpus policy.  No branch truncates content silently.
    # ------------------------------------------------------------------
    def _full_parts(self) -> tuple[list[str], list[str]]:
        parts: list[str] = []
        docnames: list[str] = []
        output_by_doc = {
            docname: path for path, docname in self._docname_by_output_file.items()
        }
        ordered_docnames = order_docnames(
            output_by_doc,
            toctree_order=self._toctree_order(),
            preferred_patterns=self._preferred_order(),
        )
        for docname in ordered_docnames:
            md_file = output_by_doc[docname]
            content = self._markdown_file_by_docname[docname].read_text(
                encoding="utf-8"
            )
            content = self._materialize_links(
                content,
                source_docname=docname,
                source_target=None,
                target_layout=None,
            )
            relative_path = md_file.relative_to(self.outdir).as_posix()
            parts.append(f"# {relative_path}\n\n{content}\n\n")
            docnames.append(docname)

        for rel in self._configured_code_files():
            source = (Path(self.app.srcdir) / rel).resolve()
            root = Path(self.app.srcdir).resolve()
            if root not in source.parents and source != root:
                raise ExtensionError(
                    f"llms_txt_code_files path escapes source root: {rel!r}"
                )
            if not source.is_file():
                raise ExtensionError(
                    f"llms_txt_code_files entry does not exist: {rel!r}"
                )
            language = source.suffix.lstrip(".") or "text"
            content = source.read_text(encoding="utf-8", errors="replace")
            parts.append(f"# Source: {rel}\n\n```{language}\n{content}\n```\n\n")
            docnames.append(f"@source/{rel}")
        return parts, docnames

    def _configured_code_files(self) -> list[str]:
        raw = getattr(self.app.config, "llms_txt_code_files", [])
        if raw is None or isinstance(raw, str):
            raise ExtensionError(
                "llms_txt_code_files must be an iterable of relative paths"
            )
        try:
            values = [normalize_relative_path(str(item)) for item in raw]
        except (TypeError, ValueError) as exc:
            raise ExtensionError(f"invalid llms_txt_code_files: {exc}") from exc
        return values

    def _full_limits(self) -> SizeLimits:
        try:
            return SizeLimits(
                max_bytes=getattr(self.app.config, "llms_txt_full_max_bytes", None),
                max_chars=getattr(self.app.config, "llms_txt_full_max_chars", None),
                max_lines=getattr(self.app.config, "llms_txt_full_max_lines", None),
                max_documents=getattr(
                    self.app.config, "llms_txt_full_max_documents", None
                ),
            )
        except ValueError as exc:
            raise ExtensionError(str(exc)) from exc

    def build_llms_full_txt(self):
        target = self.outdir / "llms-full.txt"
        parts, docnames = self._full_parts()
        policy = str(getattr(self.app.config, "llms_txt_full_size_policy", "warn_keep"))
        try:
            decision = evaluate_size_policy(parts, self._full_limits(), policy)
        except ValueError as exc:
            raise ExtensionError(str(exc)) from exc

        self._full_action = decision.action
        self._full_measure = {
            "bytes": decision.measure.bytes,
            "chars": decision.measure.chars,
            "lines": decision.measure.lines,
            "documents": decision.measure.documents,
        }
        message = "llms-full size policy exceeded: " + ", ".join(decision.exceeded)
        if decision.action == "error":
            raise ExtensionError(message)
        if decision.limited and decision.severity == "warn":
            logger.warning(
                message  # ruff: ignore[logging-string-concat]
                + f"; action={decision.action}"
            )
        elif decision.limited and decision.severity == "info":
            logger.info(
                message  # ruff: ignore[logging-string-concat]
                + f"; action={decision.action}"
            )

        if decision.action == "skip":
            target.unlink(missing_ok=True)
            self._full_included_docnames.clear()
            return
        if decision.action == "note":
            target.write_text(size_note(decision), encoding="utf-8")
            self._full_included_docnames.clear()
            return

        target.write_text("".join(parts), encoding="utf-8")
        self._full_included_docnames = {
            docname for docname in docnames if not docname.startswith("@source/")
        }
        logger.info("Generated complete llms-full.txt with %d documents", len(docnames))

    # ------------------------------------------------------------------
    # A06: deterministic, sectioned llms.txt generated from canonical pages.
    # ------------------------------------------------------------------
    def _toctree_order(self) -> dict[str, int]:
        try:
            return {
                docname: index
                for index, docname in enumerate(self.app.env.collect_relations())
            }
        except Exception:  # ruff: ignore[blind-except]
            return {}

    def _section_rules(self) -> Sequence[tuple[str, Sequence[str]]] | None:
        raw = getattr(self.app.config, "llms_txt_section_rules", [])
        if not raw:
            return None
        if isinstance(raw, (str, bytes)):
            raise ExtensionError(
                "llms_txt_section_rules must be a sequence of mappings/tuples"
            )
        rules: list[tuple[str, Sequence[str]]] = []
        try:
            for item in raw:
                if isinstance(item, dict):
                    title = str(item["title"])
                    patterns = item["patterns"]
                else:
                    title, patterns = item
                if isinstance(patterns, str):
                    patterns = [patterns]
                rules.append((str(title), tuple(str(pattern) for pattern in patterns)))
        except (KeyError, TypeError, ValueError) as exc:
            raise ExtensionError(f"invalid llms_txt_section_rules: {exc}") from exc
        return rules

    def _preferred_order(self) -> list[str]:
        raw = getattr(self.app.config, "llms_txt_order", [])
        if raw is None or isinstance(raw, str):
            raise ExtensionError("llms_txt_order must be an iterable of glob patterns")
        try:
            return [str(item) for item in raw]
        except TypeError as exc:
            raise ExtensionError("llms_txt_order must be iterable") from exc

    def _page_description_with_origin(self, md_file: Path) -> tuple[str, str]:
        docname = self._docname_by_output_file.get(md_file, "")
        if docname:
            try:
                doctree = self.app.env.get_doctree(docname)
                for node in doctree.traverse(docutils.nodes.meta):
                    content = node.get("content")
                    if (
                        node.get("name") == "description"
                        and isinstance(content, str)
                        and content.strip()
                    ):
                        return content.strip(), "author"
            except Exception:
                logger.exception("Failed to inspect author description for %s", docname)
        if docname and self._summary_enabled():
            return self.generate_page_summary(docname, md_file), "generated"
        return self.extract_description_from_markdown(md_file), "fallback"

    def _index_pages(self) -> list[IndexPage]:
        docnames = [
            self._docname_by_output_file[path] for path in self.generated_markdown_files
        ]
        ordered = order_docnames(
            docnames,
            toctree_order=self._toctree_order(),
            preferred_patterns=self._preferred_order(),
        )
        output_by_doc = {
            docname: path for path, docname in self._docname_by_output_file.items()
        }
        http_base = self._markdown_http_base()
        section_rules = self._section_rules()
        pages: list[IndexPage] = []
        for docname in ordered:
            md_file = output_by_doc[docname]
            title = self.extract_title_from_markdown(md_file)
            description, origin = self._page_description_with_origin(md_file)
            self._description_by_docname[docname] = (description, origin)
            rel = md_file.relative_to(self.outdir).as_posix()
            url = f"{http_base}/{rel}" if http_base else rel
            pages.append(
                IndexPage(
                    docname=docname,
                    title=title,
                    description=description,
                    url=url,
                    section=infer_section(docname, section_rules),
                )
            )
        return pages

    def create_sitemap(self):
        llms_txt_path = self.outdir / "llms.txt"
        http_base = self._markdown_http_base()
        full_url: str | None = None
        if (
            getattr(self.app.config, "llms_txt_full_build", True)
            and self._full_action == "keep"
        ):
            full_url = f"{http_base}/llms-full.txt" if http_base else "llms-full.txt"
        context = ""
        if hasattr(self.app.config, "copyright") and self.app.config.copyright:
            context = str(self.app.config.copyright)
        content = render_llms_index(
            project=str(getattr(self.app.config, "project", "Documentation")),
            description=self.get_project_description(),
            project_context=context,
            pages=self._index_pages(),
            full_url=full_url,
        )
        try:
            validate_text_max_bytes(
                content,
                getattr(self.app.config, "llms_txt_index_max_bytes", None),
                label="llms.txt",
            )
        except ValueError as exc:
            raise ExtensionError(
                f"{exc}; curate exclusions/sections/order rather than silently truncating the index"
            ) from exc
        llms_txt_path.write_text(content, encoding="utf-8")
        logger.info("Created deterministic semantic llms.txt index: %s", llms_txt_path)

    # ------------------------------------------------------------------
    # A08/A09/A10: manifest, compatibility, provenance, Tier-2 HTML.
    # ------------------------------------------------------------------
    def _html_path_for_docname(self, docname: str) -> str | None:
        try:
            uri = str(self.app.builder.get_target_uri(docname)).split("#", 1)[0]
        except Exception:  # ruff: ignore[blind-except]
            return None
        uri = uri.lstrip("/")
        if not uri:
            return "index.html"
        if uri.endswith("/"):
            return f"{uri}index.html"
        if Path(uri).suffix:
            return normalize_relative_path(uri)
        if getattr(self.app.builder, "name", "") == "dirhtml":
            return normalize_relative_path(f"{uri}/index.html")
        return normalize_relative_path(f"{uri}.html")

    def _markdown_paths_for_docname(self, docname: str) -> list[str]:
        targets, primary = self._target_paths_for_docname(docname)
        ordered_paths = [targets[primary]] + [
            path for layout, path in targets.items() if layout != primary
        ]
        result: list[str] = []
        for path in ordered_paths:
            rel = path.relative_to(self.outdir).as_posix()
            if rel not in result:
                result.append(rel)
        return result

    def _source_hash(self, docname: str) -> str | None:
        try:
            source = Path(self.app.env.doc2path(docname, base=True))
            if source.is_file():
                return sha256_file(source)
        except Exception:  # ruff: ignore[blind-except]
            pass
        return None

    def _description_for_manifest(self, docname: str, primary: Path) -> tuple[str, str]:
        cached = self._description_by_docname.get(docname)
        if cached is not None:
            return cached
        # Excluded pages should not trigger optional provider traffic merely for
        # diagnostics. Preserve author metadata, otherwise deterministic fallback.
        try:
            doctree = self.app.env.get_doctree(docname)
            for node in doctree.traverse(docutils.nodes.meta):
                content = node.get("content")
                if (
                    node.get("name") == "description"
                    and isinstance(content, str)
                    and content.strip()
                ):
                    result = (content.strip(), "author")
                    self._description_by_docname[docname] = result
                    return result
        except Exception:  # ruff: ignore[blind-except]
            pass
        result = (self.extract_description_from_markdown(primary), "fallback")
        self._description_by_docname[docname] = result
        return result

    def _canonical_documents(self) -> list[dict[str, Any]]:
        included_in_index = set(self._docname_by_output_file.values())
        override = str(getattr(self.app.config, "llms_txt_override_source", "")).strip()
        override_docname: str | None = None
        if override:
            normalized = override.replace("\\", "/").removeprefix("./")
            candidates = [normalized, str(Path(normalized).with_suffix(""))]
            override_docname = next(
                (item for item in candidates if item in self._markdown_file_by_docname),
                None,
            )

        records: list[dict[str, Any]] = []
        for docname in sorted(self._markdown_file_by_docname):
            markdown_paths = self._markdown_paths_for_docname(docname)
            primary = self.outdir / markdown_paths[0]
            title = self.extract_title_from_markdown(primary)
            description, origin = self._description_for_manifest(docname, primary)
            included = docname in included_in_index
            if override_docname is not None:
                included = docname == override_docname
            record = {
                "docname": docname,
                "title": title,
                "description": description or None,
                "description_origin": origin if description else "none",
                "html_path": self._html_path_for_docname(docname),
                "markdown_paths": markdown_paths,
                "source_kind": "sphinx-doctree",
                "fidelity": "canonical",
                "included_in_llms": included,
                "included_in_full": docname in self._full_included_docnames,
                "source_hash": self._source_hash(docname),
                "output_hashes": output_hashes(self.outdir, markdown_paths),
                "warnings": [],
            }
            records.append(record)
        return records

    def _fallback_target_for_html(self, html_path: str) -> str:
        return normalize_relative_path(f"{html_path}.md")

    def _build_html_fallbacks(
        self, canonical_documents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not bool(getattr(self.app.config, "llms_txt_html_fallback", True)):
            return []
        canonical_html = {
            item["html_path"] for item in canonical_documents if item.get("html_path")
        }
        canonical_markdown = {
            path
            for item in canonical_documents
            for path in item.get("markdown_paths", ())
        }
        excluded_roots = {
            "_static",
            "_sources",
            "_images",
            "_downloads",
            "_modules",
            "_llms",
        }
        records: list[dict[str, Any]] = []
        for html_file in sorted(self.outdir.rglob("*.html")):
            rel_html = html_file.relative_to(self.outdir).as_posix()
            if (
                rel_html.split("/", 1)[0] in excluded_roots
                or rel_html in canonical_html
            ):
                continue
            rel_md = self._fallback_target_for_html(rel_html)
            if rel_md in canonical_markdown:
                continue
            target = self.outdir / rel_md
            result = convert_html_file(html_file, target)
            records.append(
                {
                    "docname": f"@html/{rel_html}",
                    "title": result.title,
                    "description": result.description or None,
                    "description_origin": "fallback" if result.description else "none",
                    "html_path": rel_html,
                    "markdown_paths": [rel_md],
                    "source_kind": "post-build-html",
                    "fidelity": "compatibility",
                    "included_in_llms": False,
                    "included_in_full": False,
                    "source_hash": sha256_file(html_file),
                    "output_hashes": output_hashes(self.outdir, [rel_md]),
                    "warnings": [
                        *list(result.warnings),
                        "Tier-2 HTML compatibility conversion; not canonical",
                    ],
                }
            )
        return records

    def _generator_identity(self) -> dict[str, Any]:
        lock_path = Path(__file__).parents[1] / "sphinx_llm" / "vendor.lock.json"
        upstream_repository = None
        upstream_commit = None
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
            upstream_repository = lock.get("repository")
            upstream_commit = lock.get("commit_hash")
        except (OSError, ValueError, TypeError):
            pass
        return {
            "implementation": "scikitplot._externals._sphinx_ext._sphinx_llm",
            "version": __version__,
            "upstream_repository": upstream_repository,
            "upstream_commit": upstream_commit,
        }

    def _augmentation_record(self) -> dict[str, Any]:
        try:
            options = self._get_summary_options()
            return AugmentationSettings(
                enabled=options.enabled,
                provider=options.provider,
                model=options.model,
                base_url=options.base_url,
                api_key_env=options.api_key_env,
                allow_insecure_auth=options.allow_insecure_auth,
                max_input_chars=options.max_input_chars,
                timeout=options.timeout,
            ).public_record()
        except Exception as exc:  # ruff: ignore[blind-except]
            return {"enabled": False, "configuration_error": type(exc).__name__}

    def _compatibility_payload(self, build_id: str) -> dict[str, Any]:
        policy = str(getattr(self.app.config, "llms_txt_unknown_node_policy", "warn"))
        if self._semantic_inventory is None:
            return {
                "schema_version": 1,
                "build_id": build_id,
                "policy": policy,
                "nodes_seen": {},
                "handling": {
                    "native": 0,
                    "structural": 0,
                    "adapter": 0,
                    "media": 0,
                    "ignored_by_policy": 0,
                    "unsafe_rejected": 0,
                    "unknown": 1,
                },
                "unsupported": [
                    {
                        "node_class": "__inventory_unavailable__",
                        "documents": [],
                        "reason": (
                            "resolved-node inventory was unavailable; canonical GREEN is not established"
                        ),
                    }
                ],
                "content_loss_detected": True,
            }
        return self._semantic_inventory.compatibility_payload(
            build_id=build_id, policy=policy
        )

    def _artifact_provenance(
        self,
        manifest: dict[str, Any],
        compatibility: dict[str, Any],
    ) -> dict[str, Any]:
        build_id = manifest["build_id"]
        artifacts: list[dict[str, Any]] = []
        by_doc = manifest["documents"]
        for item in by_doc:
            for rel in item["markdown_paths"]:
                path = self.outdir / rel
                artifacts.append(
                    {
                        "path": rel,
                        "kind": "page-markdown",
                        "source_kind": item["source_kind"],
                        "fidelity": item["fidelity"],
                        "source_hash": item.get("source_hash"),
                        "output_hash": sha256_file(path) if path.is_file() else None,
                        "transforms": (
                            [
                                "resolved-sphinx-doctree",
                                "semantic-node-adapters",
                                "link-materialization",
                            ]
                            if item["fidelity"] == "canonical"
                            else [
                                "post-build-html",
                                "sanitized-compatibility-conversion",
                            ]
                        ),
                        "warnings": item.get("warnings", []),
                        "generated_by_llm": False,
                        "generation_origin": None,
                    }
                )

        generated_description = any(
            item.get("description_origin") == "generated" for item in by_doc
        )
        static_specs = [
            (
                "llms.txt",
                "llms-index",
                "sphinx-doctree",
                "canonical",
                generated_description,
            ),
            ("llms-full.txt", "llms-full", "sphinx-doctree", "canonical", False),
            (
                "_llms/manifest.json",
                "manifest",
                "artifact-contract",
                "canonical",
                False,
            ),
            (
                "_llms/compatibility.json",
                "compatibility",
                "resolved-node-inventory",
                "canonical",
                False,
            ),
        ]
        for rel, kind, source_kind, fidelity, generated_by_llm in static_specs:
            path = self.outdir / rel
            if path.is_file():
                artifacts.append(
                    {
                        "path": rel,
                        "kind": kind,
                        "source_kind": source_kind,
                        "fidelity": fidelity,
                        "source_hash": None,
                        "output_hash": sha256_file(path),
                        "transforms": ["curated-index"] if kind == "llms-index" else [],
                        "warnings": [],
                        "generated_by_llm": generated_by_llm,
                        "generation_origin": (
                            "optional build-time page-summary augmentation"
                            if generated_by_llm
                            else None
                        ),
                    }
                )
        artifacts.append(
            {
                "path": "_llms/provenance.json",
                "kind": "provenance",
                "source_kind": "artifact-contract",
                "fidelity": "canonical",
                "source_hash": None,
                "output_hash": None,
                "transforms": ["deterministic-provenance-assembly"],
                "warnings": [],
                "generated_by_llm": False,
                "generation_origin": None,
            }
        )
        return make_provenance(build_id, artifacts)

    def build_contract_artifacts(self) -> None:
        canonical = self._canonical_documents()
        fallback = self._build_html_fallbacks(canonical)
        documents = canonical + fallback
        manifest = make_manifest(
            project=str(getattr(self.app.config, "project", "Documentation")),
            docs_version=str(
                getattr(self.app.config, "release", "")
                or getattr(self.app.config, "version", "")
                or "unknown"
            ),
            language=str(getattr(self.app.config, "language", "") or "en"),
            builder=str(getattr(self.app.builder, "name", "unknown")),
            generator=self._generator_identity(),
            documents=documents,
        )
        llms_dir = self.outdir / "_llms"
        atomic_write_json(llms_dir / "manifest.json", manifest)
        compatibility = self._compatibility_payload(manifest["build_id"])
        atomic_write_json(llms_dir / "compatibility.json", compatibility)
        augmentation = self._augmentation_record()
        provenance = self._artifact_provenance(manifest, compatibility)
        provenance["augmentation"] = augmentation
        for artifact in provenance["artifacts"]:
            if artifact["kind"] == "llms-index" and augmentation.get("enabled"):
                artifact["warnings"].append(
                    "Descriptions may include optional build-time LLM summaries; "
                    "non-secret provider provenance is recorded at the provenance root"
                )
        atomic_write_json(llms_dir / "provenance.json", provenance)

    def combine_builds(self, app, exception):
        result = super().combine_builds(app, exception)
        if (
            exception is None
            and self.outdir is not None
            and self._markdown_file_by_docname
        ):
            self.build_contract_artifacts()
        return result


__all__ = ["CanonicalArtifactGenerator"]
