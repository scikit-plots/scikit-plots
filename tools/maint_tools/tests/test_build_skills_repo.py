# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tools/maint_tools/build_skills_repo.py (the released plugin, Rule 166)."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "build_skills_repo.py"
SPEC = importlib.util.spec_from_file_location("build_skills_repo", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
REPO = Path(__file__).resolve().parents[3]
#: A development build's plugin version (any valid one; the real one is the
#: checkout's ``__version__`` and HEAD commit).
DEV = "0.5.0-0.dev.0.20261115060434+g761a6ba"


def _semver_key(text):
    """Semantic Versioning 2.0.0 precedence (build metadata ignored)."""
    core, _, pre = text.split("+", 1)[0].partition("-")
    numbers = tuple(int(x) for x in core.split("."))
    if not pre:
        return numbers, (1,)
    ids = tuple((0, int(i), "") if i.isdigit() else (1, 0, i) for i in pre.split("."))
    return numbers, (0, ids)


class PluginVersionTests(unittest.TestCase):
    """The plugin version follows the library's (Rule 166)."""

    def test_the_mapping(self):
        cases = {
            "0.4.0": "0.4.0",
            "0.4.0.post11": "0.4.0+post.11",
            "0.5": "0.5.0",
            "0.5.0rc1": "0.5.0-rc.1",
            "1.2.3a4": "1.2.3-alpha.4",
            "1.2.3b0": "1.2.3-beta.0",
            "0.5.0+local.1": "0.5.0",
        }
        for library, plugin in cases.items():
            self.assertEqual(MODULE.plugin_version(library), plugin, library)
            self.assertRegex(plugin, MODULE.SEMVER)

    def test_a_development_version_carries_its_commit(self):
        # 2026-11-15 06:04:34 UTC
        got = MODULE.plugin_version("0.5.dev0", 1794722674, "761a6ba54885cd6e")
        self.assertEqual(got, "0.5.0-0.dev.0.20261115060434+g761a6ba")
        self.assertRegex(got, MODULE.SEMVER)
        later = MODULE.plugin_version("0.5.dev0", 1794722675, "0000000")
        self.assertNotEqual(got, later)  # a new commit is a new cache key
        self.assertLess(_semver_key(got), _semver_key(later))

    def test_a_development_version_needs_its_commit(self):
        for time, commit in ((None, None), (1, None), (None, "abcdef0"), (True, "abcdef0"),
                             (-1, "abcdef0"), (1, "xyz1234"), (1, "abc")):
            with self.assertRaises(ValueError):
                MODULE.plugin_version("0.5.dev0", time, commit)

    def test_versions_without_an_order_keeping_mapping_are_refused(self):
        for version in ("0.5.0rc1.dev2", "0.5.0.post1.dev0", "v0.5.0", "0.5.0-1", "", None, "abc"):
            with self.assertRaises(ValueError):
                MODULE.plugin_version(version, 1, "abcdef0")

    def test_the_order_of_library_versions_is_kept(self):
        try:
            from packaging.version import Version
        except ImportError:  # pragma: no cover - the build job has it (meson-python needs it)
            self.skipTest("packaging is not installed")
        library = ["0.4.0", "0.4.0.post1", "0.4.0.post11", "0.4.1", "0.5.dev0", "0.5.0a1",
                   "0.5.0b2", "0.5.0rc1", "0.5.0", "0.5.0.post1", "1.0.0"]
        plugin = [MODULE.plugin_version(v, 1794722674, "abcdef0") for v in library]
        by_pep440 = [plugin[i] for i in sorted(range(len(library)), key=lambda i: Version(library[i]))]
        by_semver = sorted(plugin, key=_semver_key)
        # Post releases share their release's precedence (build metadata):
        # compare on the precedence key, which is what semver defines.
        self.assertEqual([_semver_key(v) for v in by_pep440], [_semver_key(v) for v in by_semver])

    def test_the_library_version_is_read_from_the_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scikitplot").mkdir()
            (root / "scikitplot/__init__.py").write_text(
                '"""x"""\n__version__ = "0.7.dev3"  # dev\n', encoding="utf-8"
            )
            self.assertEqual(MODULE.library_version(root), "0.7.dev3")
            (root / "scikitplot/__init__.py").write_text("x = 1\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.library_version(root)

    def test_dev_version_of_this_checkout(self):
        version = MODULE.dev_version(REPO)
        self.assertRegex(version, MODULE.SEMVER)
        time, commit = MODULE.head_commit(REPO)
        self.assertTrue(version.endswith(f"+g{commit[:7]}"))
        self.assertIn("-0.dev.", version)

    def test_a_release_checkout_is_not_built_as_dev(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scikitplot").mkdir()
            (root / "scikitplot/__init__.py").write_text('__version__ = "0.5.0"\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "--version 0.5.0"):
                MODULE.dev_version(root)

    def test_head_commit_outside_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                MODULE.head_commit(Path(tmp))


class VersionTests(unittest.TestCase):
    def test_releases_are_accepted(self):
        for version in ("0.5.0", "1.10.2", "0.4.0.post11"):
            self.assertEqual(MODULE.check_version(version), version)

    def test_anything_else_is_refused(self):
        for version in ("0.5.dev0", "v0.5.0", "0.5", "0.5.0rc1", "0.5.0+git", "", None):
            with self.assertRaises(ValueError):
                MODULE.check_version(version)


class BuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files = MODULE.render(REPO, "0.5.0")

    def _built(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name)
        MODULE.apply(out, self.files)
        return out

    def test_manifest_carries_version_and_stable_repository_and_no_icon(self):
        manifest = json.loads(self.files[".claude-plugin/plugin.json"])
        self.assertEqual(manifest["version"], "0.5.0")
        self.assertEqual(
            manifest["repository"], "https://github.com/scikit-plots/skills"
        )
        # UNKNOWN_KEY_CROSS_TOOL: no `icon` key; ICON_MISSING: the file is
        # where the directory looks without it.
        self.assertNotIn("icon", manifest)
        self.assertFalse(any(rel.startswith("assets/") for rel in self.files))
        self.assertTrue(self.files[".claude-plugin/icon.png"].startswith(MODULE.PNG_MAGIC))

    def test_an_icon_is_refused_and_an_old_one_is_removed(self):
        out = self._built()
        path = out / ".claude-plugin/plugin.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**manifest, "icon": "./assets/icon.png"}), encoding="utf-8")
        self.assertTrue(any("sets 'icon'" in e for e in MODULE.validate(out)))
        (out / "assets").mkdir()
        (out / "assets/icon.png").write_bytes(b"x")
        changes = MODULE.apply(out, self.files)
        self.assertIn("- assets/icon.png", changes)
        self.assertFalse((out / "assets/icon.png").exists())
        self.assertEqual(MODULE.validate(out), [])

    def test_the_icon_file_must_be_a_square_png_of_the_listing_size(self):
        import struct

        def png(width, height, pad=0):
            return MODULE.PNG_MAGIC + b"\x00\x00\x00\x0dIHDR" + struct.pack(">II", width, height) + b"\x00" * (5 + pad)

        out = self._built()
        icon = out / ".claude-plugin/icon.png"
        cases = {
            png(800, 800): [],
            png(512, 512): [],
            png(2048, 2048): [],
            png(800, 600): ["not square"],
            png(511, 511): ["not 512 to 2048"],
            png(2049, 2049): ["not 512 to 2048"],
            b"GIF89a" + b"\x00" * 30: ["not a PNG"],
            png(800, 800, MODULE.ICON_MAX_BYTES): ["not under"],
        }
        for data, expected in cases.items():
            icon.write_bytes(data)
            errors = [e for e in MODULE.validate(out) if "icon.png" in e]
            self.assertEqual(len(errors), len(expected), (data[:30], errors))
            for message, error in zip(expected, errors):
                self.assertIn(message, error)
        icon.unlink()
        self.assertTrue(any("ICON_MISSING" in e for e in MODULE.validate(out)))

    def test_marketplace_lists_the_root_plugin_only(self):
        market = json.loads(self.files[".claude-plugin/marketplace.json"])
        self.assertEqual([p["source"] for p in market["plugins"]], ["./"])
        self.assertEqual(market["metadata"]["version"], "0.5.0")

    def test_every_skill_and_agent_of_the_plugin_is_copied(self):
        plugin = REPO / "plugins/scikitplot"
        for base in ("skills", "agents"):
            expected = {
                p.relative_to(plugin).as_posix()
                for p in (plugin / base).rglob("*")
                if p.is_file()
            }
            self.assertEqual(
                {k for k in self.files if k.startswith(base + "/")}, expected
            )

    def test_built_tree_passes_the_static_checks(self):
        self.assertEqual(MODULE.validate(self._built()), [])

    def test_build_is_idempotent_and_removes_stale_files(self):
        out = self._built()
        self.assertEqual(MODULE.diff(out, self.files), [])
        (out / "skills/scikitplot-gone").mkdir()
        (out / "skills/scikitplot-gone/SKILL.md").write_text("x", encoding="utf-8")
        (out / ".gitignore").write_text("x\n", encoding="utf-8")
        self.assertEqual(
            MODULE.apply(out, self.files),
            ["- skills/scikitplot-gone/SKILL.md", "~ .gitignore"],
        )
        self.assertFalse((out / "skills/scikitplot-gone").exists())
        self.assertEqual((out / ".gitignore").read_text(encoding="utf-8"), MODULE.GITIGNORE)

    def test_validate_reports_directory_blockers(self):
        out = self._built()
        (out / ".gitattributes").write_text("* text=auto\n", encoding="utf-8")
        (out / "README.md").write_text("Too short.\n", encoding="utf-8")
        manifest = json.loads(
            (out / ".claude-plugin/plugin.json").read_text(encoding="utf-8")
        )
        manifest["version"] = "0.5.dev0"
        (out / ".claude-plugin/plugin.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        errors = "\n".join(MODULE.validate(out))
        for needle in (".gitattributes", "README.md", "0.5.dev0"):
            self.assertIn(needle, errors)

    def test_check_command_fails_on_a_stale_target(self):
        out = self._built()
        (out / "README.md").write_text("changed\n", encoding="utf-8")
        self.assertEqual(
            MODULE.main(["check", "--out", str(out), "--version", "0.5.0"]), 1
        )
        self.assertEqual(
            MODULE.main(["build", "--out", str(out), "--version", "0.5.0"]), 0
        )
        self.assertEqual(
            MODULE.main(["check", "--out", str(out), "--version", "0.5.0"]), 0
        )




class DevBuildTests(unittest.TestCase):
    """Rule 166: main and maintenance/X.Y.x are built with a development version."""

    @classmethod
    def setUpClass(cls):
        cls.files = MODULE.render(REPO, DEV)

    def test_the_version_is_in_both_manifests(self):
        manifest = json.loads(self.files[".claude-plugin/plugin.json"])
        market = json.loads(self.files[".claude-plugin/marketplace.json"])
        self.assertEqual(manifest["version"], DEV)
        self.assertEqual(market["metadata"]["version"], DEV)

    def test_same_files_as_a_release_but_the_manifests(self):
        release = MODULE.render(REPO, "0.5.0")
        self.assertEqual(set(release), set(self.files))
        differ = {rel for rel in release if release[rel] != self.files[rel]}
        self.assertEqual(
            differ, {".claude-plugin/plugin.json", ".claude-plugin/marketplace.json"}
        )

    def test_a_tree_without_a_valid_version_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            MODULE.apply(out, self.files)
            self.assertEqual(MODULE.validate(out), [])
            path = out / ".claude-plugin/plugin.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            for bad, message in ((None, "'version' is not set"), ("0.5.dev0", "not Semantic"),
                                 ("0.5.1", "metadata.version differs")):
                if bad is None:
                    manifest.pop("version", None)
                else:
                    manifest["version"] = bad
                path.write_text(json.dumps(manifest), encoding="utf-8")
                self.assertTrue(any(message in e for e in MODULE.validate(out)), bad)

    def test_render_refuses_a_version_that_is_not_semver(self):
        for bad in (None, "0.5.dev0", "0.4.0.post11", ""):
            with self.assertRaises(ValueError):
                MODULE.render(REPO, bad)

    def test_the_command_needs_exactly_one_of_version_or_dev(self):
        with tempfile.TemporaryDirectory() as tmp:
            for argv in (["check", "--out", tmp], ["check", "--out", tmp, "--version", "0.5.0", "--dev"]):
                with self.assertRaises(SystemExit):
                    MODULE.main(argv)

    def test_the_dev_command_builds_and_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(MODULE.main(["build", "--out", tmp, "--dev"]), 0)
            self.assertEqual(MODULE.main(["check", "--out", tmp, "--dev"]), 0)
            manifest = json.loads((Path(tmp) / ".claude-plugin/plugin.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["version"], MODULE.dev_version(REPO))

    def test_a_release_is_mapped_to_semver(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(MODULE.main(["build", "--out", tmp, "--version", "0.4.0.post11"]), 0)
            manifest = json.loads((Path(tmp) / ".claude-plugin/plugin.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["version"], "0.4.0+post.11")

    def test_shared_skills_ship_in_the_published_plugin(self):
        shared = REPO / "plugins/_shared/skills"
        for path in shared.rglob("*"):
            if path.is_file():
                rel = "skills/" + path.relative_to(shared).as_posix()
                expected = path.read_bytes()
                if rel.endswith("/SKILL.md"):
                    expected = MODULE.community_component(rel, expected.decode()).encode()
                self.assertEqual(self.files[rel], expected, rel)


class PublishedTreeTests(unittest.TestCase):
    """CI-8: the directory held the plugin for a template .gitignore naming .pypirc."""

    def _built(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name)
        MODULE.apply(out, MODULE.render(REPO, "0.5.0"))
        return out

    def test_the_gitignore_names_no_credential_or_index_host(self):
        text = MODULE.GITIGNORE.lower()
        for word in (".pypirc", "pypi", "poetry", "token", "secret", ".env", "http"):
            self.assertNotIn(word, text)

    def test_an_unbuilt_top_level_file_fails(self):
        out = self._built()
        (out / "LICENSE.txt").write_text("x\n", encoding="utf-8")
        (out / ".github").mkdir()
        errors = MODULE.validate(out)
        self.assertTrue(any(e.startswith("LICENSE.txt: not built") for e in errors))
        self.assertTrue(any(e.startswith(".github: not built") for e in errors))

    def test_git_is_allowed(self):
        out = self._built()
        (out / ".git").mkdir()
        self.assertEqual(MODULE.validate(out), [])


class DirectoryReviewTests(unittest.TestCase):
    """The review held the plugin for a vendored skill-creator; it now ships none."""

    def _built(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name)
        MODULE.apply(out, MODULE.render(REPO, DEV))
        return out

    def test_no_third_party_skill_and_no_program_is_published(self):
        files = MODULE.render(REPO, DEV)
        self.assertFalse(any(rel.startswith("skills/skill-creator/") for rel in files))
        programs = [rel for rel in files if rel.endswith((".py", ".sh", ".js"))]
        self.assertEqual(programs, [])

    def test_the_published_manifest_has_no_dependencies(self):
        manifest = json.loads(MODULE.render(REPO, DEV)[".claude-plugin/plugin.json"])
        self.assertNotIn("dependencies", manifest)

    def test_a_dependency_in_the_published_manifest_fails(self):
        out = self._built()
        path = out / ".claude-plugin/plugin.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["dependencies"] = ["skill-creator@claude-plugins-official"]
        path.write_text(json.dumps(manifest), encoding="utf-8")
        errors = MODULE.validate(out)
        self.assertTrue(any("sets 'dependencies'" in e for e in errors), errors)

    def test_a_page_loading_a_remote_script_fails(self):
        out = self._built()
        for i, tag in enumerate(
            (
                '<script src="https://cdn.example.org/x.js"></script>',
                "<SCRIPT defer src='//cdn.example.org/x.js'></SCRIPT>",
                "<script src=http://cdn.example.org/x.js></script>",
            )
        ):
            page = out / f"skills/scikitplot/page{i}.html"
            page.write_text(f"<html>{tag}</html>", encoding="utf-8")
        errors = [e for e in MODULE.validate(out) if "network" in e]
        self.assertEqual(len(errors), 3, errors)

    def test_a_local_script_is_not_a_network_load(self):
        out = self._built()
        page = out / "skills/scikitplot/page.html"
        page.write_text(
            '<script src="./x.js"></script><script>1</script>', encoding="utf-8"
        )
        errors = MODULE.validate(out)
        self.assertFalse([e for e in errors if "network" in e], errors)
        # An HTML page is refused by type from a publisher that is not Verified.
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("not Markdown, text, JSON, YAML or an image", errors[0])


class CommunityPolicyTests(unittest.TestCase):
    """The directory refused 246d056 and 35a80cf/05d1ab6 for the community policy."""

    @classmethod
    def setUpClass(cls):
        cls.files = MODULE.render(REPO, DEV)

    def _built(self) -> Path:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        out = Path(tmp.name)
        MODULE.apply(out, self.files)
        return out

    def test_every_published_component_has_only_name_and_description(self):
        for rel, data in self.files.items():
            if rel.endswith("/SKILL.md") or rel.startswith("agents/"):
                head = data.decode().split("\n---\n", 1)[0].splitlines()[1:]
                self.assertEqual([line.split(":", 1)[0] for line in head], ["name", "description"], rel)

    def test_an_agent_loads_its_skill_instead_of_preloading_it(self):
        text = self.files["agents/scikitplot-corpus.md"].decode()
        self.assertNotIn("preloaded", text)
        self.assertIn("- Load the `scikitplot-corpus` skill first;", text)

    def test_every_license_text_ships_as_a_text_file(self):
        names = [rel for rel in self.files if rel.startswith("LICENSES/")]
        self.assertTrue(names)
        self.assertTrue(all(rel.endswith(".txt") for rel in names), names)
        self.assertEqual(MODULE.published_license("LICENSE_x"), "LICENSE_x.txt")
        self.assertEqual(MODULE.published_license("LICENSE_x.md"), "LICENSE_x.md")

    def test_the_built_tree_passes(self):
        self.assertEqual(MODULE.validate(self._built()), [])

    def test_each_refused_case_fails(self):
        cases = {
            "skills/scikitplot/run": (b"echo x\n", "not Markdown"),
            "skills/scikitplot/a.md": (b"#!/bin/sh\n", "starts with #!"),
            "skills/scikitplot/b.txt": (b"a\x00b", "NUL byte"),
            "skills/scikitplot/c.txt": (b"\xff\xfe", "not UTF-8"),
            "skills/scikitplot/.npmrc": (b"x\n", "configuration file"),
            "skills/scikitplot/d.md": (b"run `npm config set prefix ~/x`\n", "package source"),
            "skills/scikitplot/e.md": (b"pip install --index-url https://x y\n", "package source"),
            "skills/scikitplot/f.md": (b"writes $WORK/config.json\n", "package source"),
            "skills/scikitplot/h.md": (b"pip install -i https://pypi.example.org/simple x\n", "package source"),
            "skills/scikitplot/i.md": (b"uv pip install --index https://x y\n", "package source"),
            "skills/scikitplot/j.md": (b"pip install -f https://example.org/wheels x\n", "package source"),
            "skills/scikitplot/k.md": (b'pip install "x @ git+https://github.com/o/r.git@main#subdirectory=a"\n', "package source"),
            "skills/scikitplot/l.md": (b"pip install x@git+https://github.com/o/r.git@main\n", "package source"),
            # 246d056..5b6faa3: doctest transcripts read as writes to files.
            "skills/scikitplot/m.md": (b"```pycon\n>>> pipeline = make()\n```\n", "begins with '>'"),
            "skills/scikitplot/n.md": (b"- list item\n\n  ```\n  >>> x = 1\n  ```\n", "begins with '>'"),
            "skills/scikitplot/o.md": (b"```python\n>>> x = 1\n```\n", "does not parse"),
            "skills/scikitplot/p.md": (b"```python\npython -m scikitplot.mlflow --open-ui-seconds 5\n```\n", "does not parse"),
            "skills/scikitplot/q.md": (b"```python\nx := 1\n```\n", "does not parse"),
        }
        for rel, (data, message) in cases.items():
            with self.subTest(rel=rel):
                out = self._built()
                (out / rel).write_bytes(data)
                errors = MODULE.validate(out)
                self.assertTrue(any(rel in e and message in e for e in errors), errors)

    def test_a_name_that_only_contains_a_config_name_passes(self):
        out = self._built()
        (out / "skills/scikitplot/g.md").write_text(
            "myconfig.json and project.config.json.bak; a GitHub link"
            " https://github.com/o/r and `python -i script.py` and digit-i 3-i\n"
        )
        self.assertEqual(MODULE.validate(out), [])

    def test_code_that_runs_as_written_passes(self):
        """A prompt inside Python code, a blockquote and shell commands are not transcripts."""
        out = self._built()
        (out / "skills/scikitplot/r.md").write_text(
            "> A blockquote is prose, not code.\n\n"
            "```python\ndef f():\n    \"\"\"\n    >>> f()\n    1\n    \"\"\"\n    return 1\n"
            "f()\n# 1\n```\n\n"
            "```sh\npip install scikit-plots\npython -c \"import scikitplot\"\n```\n\n"
            "```python\nif (n := 1) > 0:\n    pass\n```\n"
        )
        self.assertEqual(MODULE.validate(out), [])

    def test_an_unclosed_fence_is_read_to_the_end(self):
        self.assertEqual(MODULE._code_blocks("a\n```sh\n> x\n"), [("sh", 3, ["> x", ""])])
        self.assertEqual(MODULE._code_blocks("````md\n```\n> x\n````\n"), [("md", 2, ["```", "> x"])])

    def test_an_extra_frontmatter_key_fails(self):
        out = self._built()
        path = out / "skills/scikitplot-corpus/SKILL.md"
        path.write_text(path.read_text().replace("\n---\n", "\nlicense: BSD-3-Clause\n---\n", 1))
        errors = MODULE.validate(out)
        self.assertTrue(any("license" in e and "frontmatter" in e for e in errors), errors)

    def test_an_agent_template_change_fails_the_build(self):
        text = "---\nname: a\ndescription: d\nskills:\n  - a\n---\n\n- Start from `a`.\n"
        with self.assertRaises(ValueError):
            MODULE.community_component("agents/a.md", text)

    def test_a_multi_line_description_fails_the_build(self):
        text = "---\nname: a\ndescription: >\n  folded\n---\nbody\n"
        with self.assertRaises(ValueError):
            MODULE.community_component("skills/a/SKILL.md", text)


if __name__ == "__main__":
    unittest.main()
