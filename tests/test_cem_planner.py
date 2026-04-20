import torch

from minidreamer.planning.cem import DiscreteCEMPlanner


class DummyWorldModel:
    def __init__(self):
        self.device = torch.device("cpu")

    def score_action_sequences(self, state, action_sequences, discount=0.99, use_done_mask=True):
        target = torch.tensor([1, 2, 0, 1], device=action_sequences.device)
        scores = -(action_sequences != target).float().sum(dim=-1)
        return {"scores": scores}


def test_discrete_cem_planner_finds_high_scoring_sequence():
    torch.manual_seed(0)
    planner = DiscreteCEMPlanner(
        world_model=DummyWorldModel(),
        action_dim=3,
        horizon=4,
        candidates=512,
        elites=64,
        iterations=5,
        discount=1.0,
        use_done_mask=False,
    )
    output = planner.plan(state=object())
    assert output.action == 1
    assert output.sequence == [1, 2, 0, 1]

