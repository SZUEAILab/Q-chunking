from pathlib import Path

from scripts.check_docs import link_targets, local_target, validate


def test_fenced_examples_do_not_create_fake_links():
    text = "[real](index.md)\n```markdown\n![example](missing.png)\n```\n"
    assert link_targets(text) == ["index.md"]


def test_image_and_reference_links_are_checked():
    assert link_targets('![curve](../images/a.png)\n[x]: other.md\n<img src="b.png">') == [
        "../images/a.png", "b.png", "other.md",
    ]


def test_relative_paths_survive_nested_docs_and_url_encoding(tmp_path):
    document = tmp_path / "docs/research/topic.md"
    assert local_target(document, "../images/a%20b.png#caption") == tmp_path / "docs/images/a b.png"
    assert local_target(document, "https://example.com/page") is None


def test_missing_links_and_unindexed_pages_are_reported(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "index.md").write_text("[broken](missing.md)\n")
    (docs / "orphan.md").write_text("# An unindexed page\n")
    errors = validate(tmp_path)
    assert any("missing missing.md" in error for error in errors)
    assert any("orphan.md: not linked" in error for error in errors)


def test_main_index_can_reach_nested_pages_without_extra_indexes(tmp_path):
    docs = tmp_path / "docs"
    current = docs / "current"
    current.mkdir(parents=True)
    (docs / "index.md").write_text("[start](current/research.md)\n")
    (current / "research.md").write_text("[next](experiments.md)\n")
    (current / "experiments.md").write_text("[back](research.md)\n")
    assert validate(tmp_path) == []


def test_repository_docs_links_and_index_coverage():
    assert validate(Path(__file__).resolve().parents[1]) == []
