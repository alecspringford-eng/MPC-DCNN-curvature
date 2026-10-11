from __future__ import annotations

import math

from dcnn_curvature.mpc.closed_loop import PlaceholderPlant


def _scaler_stats(
    x_scaler, y_index: int, control_index: int | None
) -> tuple[float, float, float, float]:
    if x_scaler is None:
        return 0.0, 1.0, 0.0, 1.0
    y_mean = float(x_scaler.x_mean[y_index])
    y_std = float(x_scaler.x_std[y_index])
    if control_index is None:
        return y_mean, y_std, 0.0, 1.0
    u_mean = float(x_scaler.x_mean[control_index])
    u_std = float(x_scaler.x_std[control_index])
    return y_mean, y_std, u_mean, u_std


class PVTOLPlant:
    def __init__(
        self,
        alpha: float = 0.1,
        ydot: float = 0.0,
        zdot: float = 0.0,
        alphadot: float = 0.0,
        delta: float = 0.5,
        g: float = 9.81,
        kp: float = 2.0,
        kd: float = 2.0,
        u_mean: float = 0.0,
        u_std: float = 1.0,
        y_mean: float = 0.0,
        y_std: float = 1.0,
    ) -> None:
        self.alpha, self.ydot, self.zdot, self.alphadot = alpha, ydot, zdot, alphadot
        self.delta, self.g, self.kp, self.kd = delta, g, kp, kd
        self.u_mean, self.u_std, self.y_mean, self.y_std = u_mean, u_std, y_mean, y_std

    def step(self, u1_std: float) -> float:
        u1 = u1_std * self.u_std + self.u_mean
        u2 = -self.kp * self.alpha - self.kd * self.alphadot
        ydd = (u1 + self.g) * math.sin(self.alpha)
        zdd = (u1 + self.g) * math.cos(self.alpha) - self.g
        self.ydot += self.delta * ydd
        self.zdot += self.delta * zdd
        self.alphadot += self.delta * u2
        self.alpha += self.delta * self.alphadot
        return float((self.alpha - self.y_mean) / self.y_std)

    @classmethod
    def from_x0(
        cls, x0, n_state_history: int, *, x_scaler=None, control_index: int | None = None, **kwargs
    ) -> PVTOLPlant:
        y_mean, y_std, u_mean, u_std = _scaler_stats(x_scaler, n_state_history - 1, control_index)
        alpha_std = float(x0[n_state_history - 1].item())
        alpha = alpha_std * y_std + y_mean
        return cls(alpha=alpha, u_mean=u_mean, u_std=u_std, y_mean=y_mean, y_std=y_std, **kwargs)


class VanDerPolPlant:
    def __init__(
        self,
        mu: float = 1.0,
        dt: float = 0.05,
        x1_init: float = 0.0,
        x2_init: float = 0.0,
        u_mean: float = 0.0,
        u_std: float = 1.0,
        y_mean: float = 0.0,
        y_std: float = 1.0,
    ) -> None:
        self.mu, self.dt = mu, dt
        self.x1, self.x2 = x1_init, x2_init
        self.u_mean, self.u_std, self.y_mean, self.y_std = u_mean, u_std, y_mean, y_std

    def _deriv(self, x1: float, x2: float, u: float) -> tuple[float, float]:
        return x2, self.mu * (1.0 - x1**2) * x2 - x1 + u

    def step(self, u_std: float) -> float:
        u = u_std * self.u_std + self.u_mean
        dt = self.dt
        k1 = self._deriv(self.x1, self.x2, u)
        k2 = self._deriv(self.x1 + dt / 2 * k1[0], self.x2 + dt / 2 * k1[1], u)
        k3 = self._deriv(self.x1 + dt / 2 * k2[0], self.x2 + dt / 2 * k2[1], u)
        k4 = self._deriv(self.x1 + dt * k3[0], self.x2 + dt * k3[1], u)
        self.x1 += dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        self.x2 += dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        return float((self.x1 - self.y_mean) / self.y_std)

    @classmethod
    def from_x0(
        cls, x0, n_state_history: int, *, x_scaler=None, control_index: int | None = None, **kwargs
    ) -> VanDerPolPlant:
        y_mean, y_std, u_mean, u_std = _scaler_stats(x_scaler, n_state_history - 1, control_index)
        x1_init = float(x0[n_state_history - 1].item()) * y_std + y_mean
        x1_prev = float(x0[n_state_history - 2].item()) * y_std + y_mean
        dt = kwargs.get("dt", 0.05)
        x2_init = (x1_init - x1_prev) / dt
        return cls(
            x1_init=x1_init,
            x2_init=x2_init,
            u_mean=u_mean,
            u_std=u_std,
            y_mean=y_mean,
            y_std=y_std,
            **kwargs,
        )


class UnicyclePlant:
    def __init__(
        self,
        theta: float = 0.1,
        ydot: float = 0.0,
        zdot: float = 0.0,
        thetadot: float = 0.0,
        delta: float = 0.5,
        kp: float = 2.0,
        kd: float = 2.0,
        u_mean: float = 0.0,
        u_std: float = 1.0,
        y_mean: float = 0.0,
        y_std: float = 1.0,
    ) -> None:
        self.theta, self.ydot, self.zdot, self.thetadot = theta, ydot, zdot, thetadot
        self.delta, self.kp, self.kd = delta, kp, kd
        self.u_mean, self.u_std, self.y_mean, self.y_std = u_mean, u_std, y_mean, y_std

    def step(self, v_std: float) -> float:
        v = v_std * self.u_std + self.u_mean
        omega = -self.kp * self.theta - self.kd * self.thetadot
        ydot_rate = v * math.sin(self.theta)
        zdot_rate = v * math.cos(self.theta)
        self.ydot += self.delta * ydot_rate
        self.zdot += self.delta * zdot_rate
        self.thetadot += self.delta * omega
        self.theta += self.delta * self.thetadot
        return float((self.theta - self.y_mean) / self.y_std)

    @classmethod
    def from_x0(
        cls, x0, n_state_history: int, *, x_scaler=None, control_index: int | None = None, **kwargs
    ) -> UnicyclePlant:
        y_mean, y_std, u_mean, u_std = _scaler_stats(x_scaler, n_state_history - 1, control_index)
        theta_std = float(x0[n_state_history - 1].item())
        theta = theta_std * y_std + y_mean
        return cls(theta=theta, u_mean=u_mean, u_std=u_std, y_mean=y_mean, y_std=y_std, **kwargs)


PLANT_REGISTRY: dict[str, type] = {
    "placeholder": PlaceholderPlant,
    "pvtol": PVTOLPlant,
    "vanderpol": VanDerPolPlant,
    "unicycle": UnicyclePlant,
}
