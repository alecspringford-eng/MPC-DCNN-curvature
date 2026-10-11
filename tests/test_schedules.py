from __future__ import annotations

import math

from dcnn_curvature.training.schedules import BetaSchedule, CurvatureDecay, TrainingSchedule


def test_staged_beta_schedule_matches_breakpoints():
    sched = BetaSchedule(
        type="staged",
        total_epochs=100,
        staged_breakpoints=(0.0, 0.2, 0.4, 0.6, 0.8),
        staged_betas=(5.0, 10.0, 20.0, 40.0, 80.0),
    )
    assert sched.beta_at(0) == 5.0
    assert sched.beta_at(20) == 10.0
    assert sched.beta_at(39) == 10.0
    assert sched.beta_at(40) == 20.0
    assert sched.beta_at(99) == 80.0


def test_multiplicative_beta_schedule_grows():
    sched = BetaSchedule(type="multiplicative", mult_beta_init=5.0, mult_r=1.1, mult_every_n_epochs=5)
    assert sched.beta_at(0) == 5.0
    assert math.isclose(sched.beta_at(5), 5.0 * 1.1)
    assert math.isclose(sched.beta_at(10), 5.0 * 1.1**2)


def test_curvature_decay_shrinks_over_epochs():
    decay = CurvatureDecay(type="exponential", c=0.1)
    values = [decay.decay_at(e) for e in (0, 10, 50, 100)]
    assert values == sorted(values, reverse=True)
    assert values[0] == 1.0


def test_curvature_decay_none_is_constant():
    decay = CurvatureDecay(type="none")
    assert decay.decay_at(0) == decay.decay_at(1000) == 1.0


def test_training_schedule_combines_lambda_h_and_decay():
    schedule = TrainingSchedule(
        beta_schedule=BetaSchedule(type="staged", total_epochs=100),
        curvature_decay=CurvatureDecay(type="exponential", c=0.05),
        lambda_h=0.02,
        lambda_beta=0.5,
    )
    state = schedule.at(0)
    assert state["beta"] == 5.0
    assert state["lambda_curvature"] == 0.02
    assert state["lambda_beta"] == 0.5

    state_later = schedule.at(20)
    assert state_later["lambda_curvature"] < 0.02
