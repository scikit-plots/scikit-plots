from __future__ import annotations

from pathlib import Path

import pytest

from scikitplot.corpus import (
    BuilderConfig,
    BuilderFactories,
    CorpusBuilder,
    DownloadPolicy,
)


def test_builder_filter_kwargs_are_applied(tmp_path: Path) -> None:
    path = tmp_path / "short.txt"
    path.write_text("x", encoding="utf-8")
    default = CorpusBuilder().build(path)
    tuned = CorpusBuilder(
        BuilderConfig(filter_kwargs={"min_words": 0, "min_chars": 0})
    ).build(path)
    assert len(default.documents) == 0
    assert len(tuned.documents) == 1


def test_builder_filter_is_cached() -> None:
    builder = CorpusBuilder(
        BuilderConfig(filter_kwargs={"min_words": 1, "min_chars": 1})
    )
    assert builder._get_filter() is builder._get_filter()


def test_make_reader_injects_builder_filter(tmp_path: Path) -> None:
    path = tmp_path / "a.txt"
    path.write_text("hello world here", encoding="utf-8")
    builder = CorpusBuilder(
        BuilderConfig(filter_kwargs={"min_words": 1, "min_chars": 1})
    )
    reader = builder._make_reader(path, chunker=None)
    assert reader.filter_ is builder._get_filter()



def test_corpus_builder_accepts_native_factories_directly(tmp_path: Path) -> None:
    created = []

    class KeepAll:
        def __call__(self, document):
            return True

    def filter_factory():
        obj = KeepAll()
        created.append(obj)
        return obj

    builder = CorpusBuilder(factories=BuilderFactories(filter_factory=filter_factory))
    assert builder._get_filter() is created[0]
    assert builder._get_filter() is created[0]


def test_corpus_builder_rejects_wrong_factory_container_type() -> None:
    try:
        CorpusBuilder(factories={"reader_factory": object()})
    except TypeError as exc:
        assert "BuilderFactories" in str(exc)
    else:  # pragma: no cover - regression guard
        raise AssertionError("CorpusBuilder accepted a non-BuilderFactories value")


def test_native_downloader_factory_receives_resolved_builder_limits(tmp_path: Path) -> None:
    captured = {}

    class DummyDownloader:
        pass

    def downloader_factory(url: str, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return DummyDownloader()

    config = BuilderConfig(
        download_timeout=7,
        max_download_bytes=1234,
        download_max_retries=2,
        download_retry_backoff=0.25,
    )
    builder = CorpusBuilder(
        config,
        factories=BuilderFactories(downloader_factory=downloader_factory),
    )
    downloader = builder._make_downloader("https://example.com/file.txt")

    assert isinstance(downloader, DummyDownloader)
    assert captured["url"] == "https://example.com/file.txt"
    assert captured["timeout"] == 7
    assert captured["max_bytes"] == 1234
    assert captured["max_retries"] == 2
    assert captured["retry_backoff"] == 0.25
    assert Path(captured["output_path"]).is_dir()
    assert str(Path(captured["output_path"])).startswith(str(builder._get_temp_dir()))
    builder.close()


def test_builder_downloader_kwargs_tune_builtin_without_custom_factory() -> None:
    builder = CorpusBuilder(
        BuilderConfig(
            downloader_kwargs={
                "verify_ssl": False,
                "block_private_ips": False,
                "max_redirects": 2,
                "youtube_language": "tr",
            }
        )
    )
    downloader = builder._make_downloader("https://example.com/file.txt")
    assert downloader.verify_ssl is False
    assert downloader.block_private_ips is False
    assert downloader.max_redirects == 2
    assert downloader.youtube_language == "tr"
    builder.close()


def test_builder_callsite_downloader_kwargs_override_config_mapping() -> None:
    builder = CorpusBuilder(
        BuilderConfig(downloader_kwargs={"verify_ssl": False, "max_redirects": 1})
    )
    downloader = builder._make_downloader(
        "https://example.com/file.txt", verify_ssl=True, max_redirects=4
    )
    assert downloader.verify_ssl is True
    assert downloader.max_redirects == 4
    builder.close()


def test_builder_download_policy_is_authoritative_until_explicit_override() -> None:
    builder = CorpusBuilder(
        BuilderConfig(
            download_policy=DownloadPolicy.constrained(),
            downloader_kwargs={"max_redirects": 1},
        )
    )
    downloader = builder._make_downloader("https://example.com/file.txt")
    assert downloader.timeout == 15.0
    assert downloader.max_bytes == 25 * 1024 * 1024
    assert downloader.max_redirects == 1
    assert downloader.verify_ssl is True
    assert downloader.block_private_ips is True
    builder.close()


def test_builder_download_policy_rejects_ambiguous_legacy_overrides() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        BuilderConfig(
            download_policy=DownloadPolicy.constrained(),
            download_timeout=999,
        )


def test_builder_download_policy_allows_deliberate_downloader_kwargs_override() -> None:
    builder = CorpusBuilder(
        BuilderConfig(
            download_policy=DownloadPolicy.constrained(),
            downloader_kwargs={"timeout": 8, "max_bytes": 1234},
        )
    )
    downloader = builder._make_downloader("https://example.com/file.txt")
    assert downloader.timeout == 8.0
    assert downloader.max_bytes == 1234
    builder.close()


def test_builder_download_policy_mapping_is_supported() -> None:
    builder = CorpusBuilder(
        BuilderConfig(
            download_policy={
                "preset": "secure",
                "timeout": 9,
                "max_bytes": 12345,
            }
        )
    )
    downloader = builder._make_downloader("https://example.com/file.txt")
    assert downloader.timeout == 9.0
    assert downloader.max_bytes == 12345
    builder.close()
