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


def test_theme_progress_rejects_out_of_order_completion(tmp_path):
    progress = batch.ThemeProgress(
        tmp_path, "theme", {"base": 1, "usdu": 2, "seedvr2": 3}
    )
    with pytest.raises(ValueError, match="expected 0"):
        progress.mark_completed(1)
