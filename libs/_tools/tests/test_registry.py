# libs/_tools/tests/test_registry.py
#
# Authors: The scikit-plots developers
# SPDX-License-Identifier: BSD-3-Clause

"""Tests of ``libs/_tools/registry.py``: the packaging facts are coherent."""

from __future__ import annotations

import pytest

from .. import generate, registry, staging

ROOT = staging.repo_root()
MAP = staging.load_distributions(ROOT)
META = generate.read_root_metadata(ROOT)
PACKAGES = registry.by_distribution()


class TestDirectoryOf:
    @pytest.mark.parametrize(
        ("name", "directory"),
        [
            ("scikit-plots-skinny", "skinny"),
            ("scikit-plots-rank-bm25", "rank-bm25"),
            ("scikit-plots-corpus", "corpus"),
        ],
    )
    def test_directory_is_the_name_without_the_prefix(self, name, directory):
        assert registry.directory_of(name) == directory

    @pytest.mark.parametrize("bad", ["scikit-plots", "scikit-plots-", "corpus", "scikit_plots_corpus"])
    def test_a_name_without_the_prefix_is_refused(self, bad):
        with pytest.raises(ValueError, match="not of the form"):
            registry.directory_of(bad)

    def test_every_directory_exists(self):
        for name in PACKAGES:
            assert (ROOT / "libs" / registry.directory_of(name)).is_dir(), name


class TestAgreementWithTheDistributionMap:
    def test_same_distributions_in_the_same_order(self):
        assert list(PACKAGES) == [dist.name for dist in MAP.DISTRIBUTIONS]

    def test_a_distribution_declared_twice_is_refused(self, monkeypatch):
        monkeypatch.setattr(registry, "PACKAGES", registry.PACKAGES + registry.PACKAGES[:1])
        with pytest.raises(ValueError, match="declared twice"):
            registry.by_distribution()


@pytest.mark.parametrize("package", registry.PACKAGES, ids=lambda p: p.distribution)
class TestEveryPackage:
    def test_licence_is_stated(self, package):
        assert package.license.strip() == package.license != ""
        assert "BSD-3-Clause" in package.license

    def test_inherited_dependencies_are_declared_by_the_root(self, package):
        lines = generate.inherit_requirements(package.inherit, META.dependencies)
        assert len(lines) >= len(package.inherit)

    def test_written_out_requirements_are_neither_pinned_nor_capped(self, package):
        # The project declares requirements version-free: a lower bound is
        # allowed where a failure was measured, a pin or an upper bound is not.
        for requirement in package.requires:
            for operator in ("==", "<", "~=", "!="):
                assert operator not in requirement, f"{requirement}: {operator}"

    def test_written_out_requirements_are_not_in_the_root(self, package):
        # A requirement the root declares must be inherited, not restated.
        declared = {generate.requirement_name(line) for line in META.dependencies}
        for requirement in package.requires:
            assert generate.requirement_name(requirement) not in declared, requirement

    def test_lowest_constraints_narrow_a_dependency_the_part_has(self, package):
        # A test-only constraint on something the part does not depend on, or
        # one that pins or caps, is a mistake or a leftover.
        lines = generate.inherit_requirements(package.inherit, META.dependencies)
        names = {generate.requirement_name(line) for line in lines + list(package.requires)}
        for constraint in package.lowest_constraints:
            assert generate.requirement_name(constraint) in names, constraint
            assert ">=" in constraint, constraint
            for operator in ("==", "<", "~=", "!="):
                assert operator not in constraint, f"{constraint}: {operator}"

    def test_ignored_tests_exist_inside_an_owned_tree(self, package):
        # An entry that names no file, or a file of another distribution,
        # silently ignores nothing.
        trees = MAP.get(package.distribution).trees
        for entry in package.test_ignore:
            assert any(entry.startswith(tree + "/") for tree in trees), entry
            assert (ROOT / "scikitplot" / entry).exists(), entry

    def test_a_test_floor_is_above_the_install_floor(self, package):
        # ``test_python`` equal to or below ``requires-python`` says nothing.
        if package.test_python is None:
            return
        floor = package.requires_python or META.requires_python
        assert package.test_python.startswith(">=")
        lowest_supported = floor[2:].strip()
        assert not registry.python_satisfies(package.test_python, lowest_supported)

    def test_extras_exist_in_the_root(self, package):
        for extra in package.extras:
            assert extra in META.extras, extra

    def test_siblings_are_other_partial_distributions(self, package):
        for extra, siblings in package.siblings:
            assert extra not in package.extras
            for sibling in siblings:
                assert sibling in PACKAGES and sibling != package.distribution
                assert sibling != MAP.CORE, "the core is already a dependency"

    def test_test_extras_are_its_own_extras(self, package):
        own = set(package.extras) | {extra for extra, _ in package.siblings}
        assert set(package.test_extras) <= own

    def test_example_is_stated_and_its_language_is_known(self, package):
        assert package.example.endswith("\n") and package.example.strip()
        assert package.example_language in {"python", "shell"}

    def test_extension_paths_are_owned_by_the_distribution(self, package):
        owned = {
            f"scikitplot/{path.as_posix()}"
            for path in staging.iter_owned_files(
                MAP.get(package.distribution), ROOT / "scikitplot"
            )
        }
        for extension in package.extensions:
            assert extension.name.startswith("scikitplot.")
            assert MAP.provider_of(extension.name) == package.distribution
            for template in extension.templates:
                assert template.endswith(".in") and template in owned, template
            generated = {template[: -len(".in")] for template in extension.templates}
            for source in extension.sources:
                assert source in owned or source in generated, source
            for include in extension.include_dirs:
                assert any(path.startswith(include + "/") for path in owned), include


class TestWholeRegistry:
    def test_only_the_core_installs_the_console_script(self):
        assert [p.distribution for p in registry.PACKAGES if p.scripts] == [MAP.CORE]

    def test_the_core_has_no_dependencies(self):
        core = PACKAGES[MAP.CORE]
        assert (core.inherit, core.requires, core.extras, core.siblings) == ((), (), (), ())

    def test_sibling_extras_are_acyclic(self):
        edges = {
            p.distribution: {s for _, siblings in p.siblings for s in siblings}
            for p in registry.PACKAGES
        }

        def reaches(start, target, seen=()):
            return any(
                nxt == target or (nxt not in seen and reaches(nxt, target, (*seen, nxt)))
                for nxt in edges[start]
            )

        for name in edges:
            assert not reaches(name, name), f"{name} reaches itself through sibling extras"

    def test_build_requirement_has_a_floor_and_no_pin(self):
        for requirement in registry.BUILD_REQUIRES:
            assert ">=" in requirement and "==" not in requirement

    def test_compiled_distributions_declare_what_translates_them(self):
        for package in registry.PACKAGES:
            needs_cython = any(
                source.endswith(".pyx") or ext.templates
                for ext in package.extensions
                for source in ext.sources
            )
            assert ("cython" in package.build_inherit) == needs_cython, package.distribution

    def test_cleanprompt_extras_name_the_tiers_the_submodule_checks(self):
        """
        Each tier extra installs exactly the distribution its tier checks.

        ``scikitplot/cleanprompt/_capabilities.py`` holds the table the
        submodule checks installed versions against, and it is the one place
        that states supported versions: the extras are version-free. The table
        is read as text: the tooling never imports the package.
        """
        import ast

        source = (ROOT / "scikitplot" / "cleanprompt" / "_capabilities.py").read_text(
            encoding="utf-8"
        )
        tiers = {}
        for node in ast.walk(ast.parse(source)):
            if not (isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "TIERS"):
                continue
            for key, call in zip(node.value.keys, node.value.values):
                fields = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
                tiers[key.value] = fields["distribution"]
        assert tiers, "TIERS table not found in cleanprompt/_capabilities.py"
        for tier, distribution in tiers.items():
            assert META.extras[f"cleanprompt-{tier}"] == (distribution,), tier
            assert distribution in META.extras["cleanprompt"], tier


class TestPythonSatisfies:
    @pytest.mark.parametrize(
        ("specifier", "version", "expected"),
        [
            (">=3.8", "3.8", True),
            (">=3.8", "3.14", True),
            (">=3.10", "3.9", False),  # compared as numbers, not as text
            (">=3.10", "3.10", True),
            (">=3.11", "3.10.20", False),
            (">=3.11", "3.11.0", True),
            (">= 3.9", "3.9", True),
            (">=3.9", "4.0", True),
        ],
    )
    def test_comparison(self, specifier, version, expected):
        assert registry.python_satisfies(specifier, version) is expected

    @pytest.mark.parametrize("specifier", ["3.8", ">3.8", ">=3", ">=3.8,<4", "~=3.8", "", ">=3.8.1"])
    def test_any_other_form_is_refused(self, specifier):
        with pytest.raises(ValueError, match="not of the form '>=X.Y'"):
            registry.python_satisfies(specifier, "3.12")

    @pytest.mark.parametrize("version", ["3", "three.twelve", "", "3.x"])
    def test_a_malformed_version_is_refused(self, version):
        with pytest.raises(ValueError, match="not a Python version"):
            registry.python_satisfies(">=3.8", version)


class TestPythonFloors:
    def test_the_root_floor_has_the_supported_form(self):
        assert registry.python_satisfies(META.requires_python, "3.99") is True

    @pytest.mark.parametrize("package", registry.PACKAGES, ids=lambda p: p.distribution)
    def test_an_override_only_narrows_the_root_range(self, package):
        if package.requires_python is None:
            return
        floor = package.requires_python.replace(">=", "")
        # The part's own floor must itself be allowed by the root project.
        assert registry.python_satisfies(META.requires_python, floor)
        assert package.requires_python != META.requires_python, "restates the root; remove it"

    @pytest.mark.parametrize("package", registry.PACKAGES, ids=lambda p: p.distribution)
    def test_test_extras_floor_is_narrower_than_the_distribution_floor(self, package):
        if package.test_extras_python is None:
            return
        assert package.test_extras, "a floor for test extras that do not exist"
        own = (package.requires_python or META.requires_python).replace(">=", "")
        assert not registry.python_satisfies(package.test_extras_python, own), (
            "the test extras are installable wherever the distribution is; remove the floor"
        )

    def test_the_core_supports_every_python_the_root_does(self):
        # Every other distribution depends on the core, so the core can never
        # be the one that narrows the range.
        assert PACKAGES[MAP.CORE].requires_python is None


class TestPythonGated:
    @pytest.mark.parametrize("package", registry.PACKAGES, ids=lambda p: p.distribution)
    def test_gated_modules_are_owned_and_exist(self, package):
        own = (package.requires_python or META.requires_python).replace(">=", "")
        for module, floor in package.python_gated:
            assert MAP.provider_of(module) == package.distribution, module
            relative = module.split(".", 1)[1].replace(".", "/")
            path = ROOT / "scikitplot" / relative
            assert path.with_suffix(".py").is_file() or (path / "__init__.py").is_file(), module
            # A gate at or below the distribution's own floor gates nothing.
            assert not registry.python_satisfies(floor, own), (module, floor)
