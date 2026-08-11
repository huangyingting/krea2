"""Prompt-file / folder iteration and the resume log."""

from __future__ import annotations

import pytest

from krea2pipe import batch


def test_iter_prompts_skips_blank_and_comment_lines(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("first\n\n# a comment\n  second  \n")
    prompts = list(batch.iter_prompts(f))
    assert [(p.line, p.text) for p in prompts] == [(1, "first"), (4, "second")]


def test_iter_prompts_walks_a_folder_in_a_stable_order(tmp_path):
    (tmp_path / "b.txt").write_text("bravo\n")
    (tmp_path / "a.txt").write_text("alpha\n")
    (tmp_path / "notes.md").write_text("ignored\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.prompt").write_text("charlie\n")
    assert [p.text for p in batch.iter_prompts(tmp_path)] == ["alpha", "bravo", "charlie"]


def test_iter_prompts_rejects_a_folder_without_prompt_files(tmp_path):
    (tmp_path / "image.png").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        list(batch.iter_prompts(tmp_path))


def test_progress_survives_a_restart(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("one\ntwo\nthree\n")
    prompts = list(batch.iter_prompts(f))

    progress = batch.Progress(tmp_path)
    assert [p for p in prompts if p not in progress] == prompts
    progress.mark(prompts[0])
    progress.mark(prompts[1])

    # A fresh process reads the log back and only the last line is left to do.
    resumed = batch.Progress(tmp_path)
    assert [p.text for p in prompts if p not in resumed] == ["three"]


def test_progress_marking_twice_writes_one_record(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("one\n")
    prompt = next(iter(batch.iter_prompts(f)))
    progress = batch.Progress(tmp_path)
    progress.mark(prompt)
    progress.mark(prompt)
    assert progress.path.read_text().count("\n") == 1


def test_progress_distinguishes_files_with_equal_lines(tmp_path):
    (tmp_path / "a.txt").write_text("same\n")
    (tmp_path / "b.txt").write_text("same\n")
    first, second = batch.iter_prompts(tmp_path)
    progress = batch.Progress(tmp_path)
    progress.mark(first)
    assert first in progress and second not in progress
    assert first.seed_offset != second.seed_offset


def test_source_queue_indexes_a_large_tree_incrementally(tmp_path, monkeypatch):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    for file_index in range(100):
        (source / f"{file_index:03}.txt").write_text(
            "".join(f"prompt {file_index}-{line}\n" for line in range(100))
        )

    with batch.SourceQueue(source, output) as queue:
        assert queue.reconcile() == 100
        assert queue.counts() == (10_000, 0, 10_000)

        original_open = batch.Path.open

        def reject_prompt_reread(path, *args, **kwargs):
            if path.suffix == ".txt":
                raise AssertionError(f"unchanged prompt file was reread: {path}")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(batch.Path, "open", reject_prompt_reread)
        assert queue.reconcile() == 0
        assert queue.counts() == (10_000, 0, 10_000)


def test_source_queue_combines_files_and_folders(tmp_path):
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "a.txt").write_text("alpha\n")
    file = tmp_path / "standalone.prompt"
    file.write_text("bravo\n")

    with batch.SourceQueue([folder, file], tmp_path / "output") as queue:
        queue.reconcile()
        assert queue.counts() == (2, 0, 2)
        assert queue.next_pending().text == "alpha"


def test_source_queue_accepts_an_explicit_file_with_any_extension(tmp_path):
    file = tmp_path / "prompts.data"
    file.write_text("explicit prompt\n")

    with batch.SourceQueue(file, tmp_path / "output") as queue:
        queue.reconcile()
        assert queue.counts() == (1, 0, 1)


def test_source_spec_canonicalizes_symlinked_inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    prompt = source / "prompt.txt"
    prompt.write_text("prompt\n")
    link = tmp_path / "linked-source"
    link.symlink_to(source, target_is_directory=True)

    spec = batch.SourceSpec([f"{link}/**/*.txt"])

    assert list(spec.iter_files()) == [prompt]
    assert spec.watch_roots == (source,)


def test_source_queue_applies_unified_include_and_exclude_globs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "keep.txt").write_text("keep\n")
    (source / "draft-a.txt").write_text("draft\n")
    archive = source / "archive"
    archive.mkdir()
    (archive / "old.txt").write_text("old\n")
    nested = source / "nested"
    nested.mkdir()
    (nested / "photo-1.data").write_text("photo\n")
    (nested / "photo-x.data").write_text("wrong class\n")

    sources = [
        f"{source}/**/*.txt",
        f"{source}/**/photo-[0-9].data",
        f"!{source}/archive/**",
        f"!{source}/**/draft-?.txt",
    ]
    with batch.SourceQueue(sources, tmp_path / "output") as queue:
        queue.reconcile()
        assert queue.counts() == (2, 0, 2)
        assert queue.next_pending().text == "keep"


def test_source_spec_resolves_relative_globs_and_derives_watch_roots(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    archive = source / "archive"
    archive.mkdir()
    keep = source / "keep.txt"
    keep.write_text("keep\n")
    (archive / "old.txt").write_text("old\n")
    monkeypatch.chdir(tmp_path)

    spec = batch.SourceSpec(["source/**/*.txt", "!source/archive/**"])

    assert list(spec.iter_files()) == [keep]
    assert spec.watch_roots == (source,)
    assert spec.identity == (
        str(source / "**" / "*.txt"),
        f"!{source / 'archive' / '**'}",
    )


def test_source_glob_star_does_not_cross_directory_boundaries(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    top = source / "top.txt"
    top.write_text("top\n")
    nested = source / "nested"
    nested.mkdir()
    (nested / "nested.txt").write_text("nested\n")

    spec = batch.SourceSpec([f"{source}/*.txt"])

    assert list(spec.iter_files()) == [top]


def test_source_spec_requires_a_positive_existing_root(tmp_path):
    with pytest.raises(ValueError, match="at least one positive"):
        batch.SourceSpec([f"!{tmp_path}/archive/**"])
    with pytest.raises(FileNotFoundError, match="source glob root"):
        batch.SourceSpec([f"{tmp_path}/missing/**/*.txt"])


def test_source_queue_indexes_only_appended_lines(tmp_path):
    source = tmp_path / "prompts.txt"
    output = tmp_path / "output"
    source.write_text("one\ntwo\n")

    with batch.SourceQueue(source, output) as queue:
        queue.reconcile()
        first = queue.next_pending()
        queue.mark(first)
        second = queue.next_pending()
        queue.mark(second)
        source.write_text("one\ntwo\nthree\n")
        assert queue.update_paths({source}) == 1
        assert queue.counts() == (3, 2, 1)
        assert queue.next_pending().text == "three"


def test_source_queue_reindexes_only_a_changed_file(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    first = source / "first.txt"
    second = source / "second.txt"
    first.write_text("keep\nold\n")
    second.write_text("untouched\n")

    with batch.SourceQueue(source, output) as queue:
        queue.reconcile()
        queue.mark(queue.next_pending())
        first.write_text("keep\nnew\n")
        assert queue.update_paths({first}) == 1
        assert queue.counts() == (3, 1, 2)
        assert [queue.next_pending().text] == ["new"]


def test_source_queue_removes_deleted_pending_files(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    file = source / "pending.txt"
    file.write_text("pending\n")

    with batch.SourceQueue(source, output) as queue:
        queue.reconcile()
        file.unlink()
        assert queue.update_paths({file}) == 1
        assert queue.counts() == (0, 0, 0)


def test_source_queue_survives_restart_and_imports_legacy_progress(tmp_path):
    source = tmp_path / "prompts.txt"
    output = tmp_path / "output"
    source.write_text("one\ntwo\n")
    prompts = list(batch.iter_prompts(source))
    legacy = batch.Progress(output)
    legacy.mark(prompts[0])

    with batch.SourceQueue(source, output) as queue:
        queue.reconcile()
        assert queue.counts() == (2, 1, 1)
        queue.mark(queue.next_pending())

    with batch.SourceQueue(source, output) as resumed:
        resumed.reconcile()
        assert resumed.counts() == (2, 2, 0)


def test_source_queue_reset_clears_active_completion_state(tmp_path):
    source = tmp_path / "prompts.txt"
    source.write_text("one\ntwo\n")
    output = tmp_path / "output"

    with batch.SourceQueue(source, output) as queue:
        queue.reconcile()
        queue.mark(queue.next_pending())
        assert queue.counts() == (2, 1, 1)
        assert queue.reset() == 1
        assert queue.counts() == (2, 0, 2)


def test_source_watcher_reports_new_prompt_files(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    with batch.SourceWatcher(source) as watcher:
        file = source / "new.txt"
        file.write_text("new prompt\n")
        changed = watcher.wait(5)

    assert file.resolve() in {path.resolve() for path in changed}


def test_source_watcher_incrementally_updates_queue(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "output"

    spec = batch.SourceSpec([f"{source}/**/*.prompts"])
    with (
        batch.SourceQueue(spec, output) as queue,
        batch.SourceWatcher(spec) as watcher,
    ):
        queue.reconcile()
        ignored = source / "ignored.txt"
        ignored.write_text("ignored prompt\n")
        file = source / "new.prompts"
        file.write_text("new prompt\n")
        queue.update_paths(watcher.wait(5))
        assert queue.counts() == (1, 0, 1)
        assert queue.next_pending().text == "new prompt"


def test_seed_offset_is_stable(tmp_path):
    file = tmp_path / "p.txt"
    file.write_text("one\n")
    first = next(batch.iter_prompts(file))
    second = next(batch.iter_prompts(file))
    assert first.seed_offset == second.seed_offset


def test_output_lock_rejects_a_concurrent_renderer(tmp_path):
    with batch.OutputLock(tmp_path):
        with pytest.raises(batch.AlreadyRunningError, match="another krea2pipe process"):
            with batch.OutputLock(tmp_path):
                pass

    # Releasing the first lock makes the output directory available again.
    with batch.OutputLock(tmp_path):
        pass


def test_theme_progress_persists_index_and_original_seeds(tmp_path):
    seeds = {"base": 11, "usdu": 22, "seedvr2": 33}
    progress = batch.ThemeProgress(tmp_path, "calm ocean", seeds)
    assert progress.next_index == 0
    assert progress.seeds == seeds
    progress.mark_completed(0)
    progress.mark_completed(1)

    resumed = batch.ThemeProgress(
        tmp_path,
        "calm ocean",
        {"base": 999, "usdu": 999, "seedvr2": 999},
    )
    assert resumed.next_index == 2
    assert resumed.seeds == seeds


def test_theme_progress_tracks_themes_independently(tmp_path):
    first = batch.ThemeProgress(
        tmp_path, "ocean", {"base": 1, "usdu": 2, "seedvr2": 3}
    )
    first.mark_completed(0)
    second = batch.ThemeProgress(
        tmp_path, "forest", {"base": 4, "usdu": 5, "seedvr2": 6}
    )
    assert second.next_index == 0
    assert batch.ThemeProgress(
        tmp_path, "ocean", {"base": 7, "usdu": 8, "seedvr2": 9}
    ).next_index == 1


def test_theme_progress_tracks_system_prompts_independently(tmp_path):
    seeds = {"base": 1, "usdu": 2, "seedvr2": 3}
    first = batch.ThemeProgress(tmp_path, "ocean", seeds, "Use Chinese.")
    first.mark_completed(0)

    second = batch.ThemeProgress(tmp_path, "ocean", seeds, "Use English.")

    assert second.next_index == 0
    assert batch.ThemeProgress(
        tmp_path, "ocean", seeds, "Use Chinese."
    ).next_index == 1


def test_theme_progress_rejects_out_of_order_completion(tmp_path):
    progress = batch.ThemeProgress(
        tmp_path, "theme", {"base": 1, "usdu": 2, "seedvr2": 3}
    )
    with pytest.raises(ValueError, match="expected 0"):
        progress.mark_completed(1)


def test_theme_progress_reset_starts_sequence_over(tmp_path):
    seeds = {"base": 1, "usdu": 2, "seedvr2": 3}
    progress = batch.ThemeProgress(tmp_path, "theme", seeds)
    progress.mark_completed(0)
    assert progress.reset() == 1

    restarted = batch.ThemeProgress(tmp_path, "theme", seeds)
    assert restarted.next_index == 0
