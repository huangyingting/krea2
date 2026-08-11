import pytest
import torch

from krea2pipe import workflow


def test_prefetch_retains_and_reraises_worker_exception():
    error = RuntimeError("prefetch failed")
    calls = 0

    def fail():
        nonlocal calls
        calls += 1
        raise error

    wait = workflow._prefetch(fail)

    for _ in range(2):
        with pytest.raises(RuntimeError, match="prefetch failed") as raised:
            wait()
        assert raised.value is error

    assert calls == 1


def test_workflow_does_not_sample_after_prefetch_failure(monkeypatch):
    class FailingPipeline:
        sample_called = False

        @property
        def dit(self):
            raise RuntimeError("LoRA load failed")

        def encode_prompt(self, _prompt):
            return torch.zeros(1)

        def sample(self, *_args, **_kwargs):
            self.sample_called = True
            raise AssertionError("sampling must not start")

    pipe = FailingPipeline()
    monkeypatch.setattr(workflow, "preflight", lambda _cfg: None)
    monkeypatch.setattr(workflow.accel, "tune_backends", lambda: None)
    monkeypatch.setattr(workflow, "_cached_pipeline", lambda _cfg: pipe)
    cfg = workflow.WorkflowConfig(
        width=32,
        height=32,
        device="cpu",
        dtype=torch.float32,
        save=False,
        run_usdu=False,
        run_color_match=False,
        run_seedvr2=False,
        run_blend=False,
    )

    with pytest.raises(RuntimeError, match="LoRA load failed"):
        workflow.run_workflow(cfg)

    assert pipe.sample_called is False
