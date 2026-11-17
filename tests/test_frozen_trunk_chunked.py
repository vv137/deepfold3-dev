"""Chunked diffusion training with a frozen trunk (``freeze_modules``).

The chunked path detaches the trunk outputs into leaves, back-props each
diffusion chunk into them, then flows the leaf grads back through the trunk.
With the trunk frozen the trunk outputs carry no graph, and that last walk
raised "element 0 of tensors does not require grad" on the first step of the
stage3-s1-frozen-cfg runs.
"""

from contextlib import nullcontext

import pytest
import torch

FROZEN = ["evo_cond_encoder", "evoformer", "confidence_head", "distogram_head"]


class _Acc:
    def chunk_context(self, **kw):
        return nullcontext()


def _cfg(chunk: int):
    from deepfold3.model.config import TrainingConfig

    return TrainingConfig(
        batch_size_per_rank=1,
        diffusion_batch_size=2,
        diffusion_chunk_size=chunk,
        mini_rollout_steps=2,
        alpha_confidence=0.0,
        alpha_distogram=0.0,
        alpha_bond=1.0,
        max_recycles=1,
    )


def _aux_backward(cfg, loss):
    from types import SimpleNamespace

    from deepfold3.training.trainer import Trainer

    fake = SimpleNamespace(cfg=cfg, scaler=torch.amp.GradScaler("cpu", enabled=False))
    fake._chunked_diffusion_active = lambda: Trainer._chunked_diffusion_active(fake)
    Trainer._scaled_backward(fake, loss)


def test_unchunked_graphless_loss_still_raises():
    with pytest.raises(RuntimeError, match="does not require grad"):
        _aux_backward(_cfg(0), torch.tensor(1.0))


@pytest.mark.parametrize("chunk", [0, 1])
@pytest.mark.parametrize("fill", ["zero", "learned"])
def test_frozen_trunk_training_step_backprops_into_the_diffusion_head_only(chunk, fill):
    import deepfold3.training.step as step
    from deepfold3.training.optim import freeze_modules
    from tests.test_diffusion_guidance import _tiny_batch, _tiny_model

    model = _tiny_model(use_conditioning_prob=0.5, cond_drop_fill=fill).train()
    freeze_modules(model, FROZEN)
    cfg = _cfg(chunk)
    if chunk:
        assert step.is_chunked_diffusion_active(cfg)
    torch.manual_seed(0)
    loss, *_ = step.training_step(
        model.forward_training,
        _tiny_batch(),
        cfg=cfg,
        model_cfg=model.cfg,
        global_step=0,
        accumulator=_Acc(),
        scaler=torch.amp.GradScaler("cpu", enabled=False),
        device=torch.device("cpu"),
    )
    # The trainer's aux backward on the returned loss.
    _aux_backward(cfg, loss)

    for name in FROZEN:
        assert all(p.grad is None for p in getattr(model, name).parameters()), name
    grads = [p.grad for p in model.diffusion_head.parameters() if p.requires_grad]
    assert any(g is not None and bool(g.abs().sum() > 0) for g in grads)
