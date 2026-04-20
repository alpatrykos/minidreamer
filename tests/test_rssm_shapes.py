import torch

from minidreamer.models.world_model import WorldModel


def test_world_model_sequence_shapes_and_loss():
    torch.manual_seed(0)
    model = WorldModel(
        action_dim=7,
        embedding_dim=128,
        deter_dim=128,
        stoch_dim=16,
        hidden_dim=128,
        use_decoder=True,
    )
    obs = torch.rand(4, 33, 3, 64, 64)
    actions = torch.randint(0, 7, (4, 32))
    outputs = model.observe_sequence(obs, actions, sample=False)
    assert outputs.reward_pred.shape == (4, 32)
    assert outputs.done_logits.shape == (4, 32)
    assert outputs.prior_mean.shape == (4, 32, 16)
    assert outputs.reconstructions is not None
    assert outputs.reconstructions.shape == (4, 32, 3, 64, 64)

    batch = {
        "obs": obs,
        "actions": actions,
        "rewards": torch.zeros(4, 32),
        "done": torch.zeros(4, 32),
        "mask": torch.ones(4, 32),
    }
    config = {
        "training": {
            "beta_reward": 1.0,
            "beta_done": 1.0,
            "beta_kl": 1.0,
            "beta_recon": 1.0,
            "free_nats": 1.0,
        }
    }
    losses = model.compute_losses(batch, config)
    assert torch.isfinite(losses["loss"])

