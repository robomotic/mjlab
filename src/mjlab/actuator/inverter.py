"""Inverter model for AC motor drives (PMSM).

This module provides DC-to-AC inverter efficiency modeling for Permanent
Magnet Synchronous Motors (PMSM) commonly used in modern robots like Unitree
Go2 and H1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class InverterCfg:
  """Configuration for DC-to-AC inverter for PMSM motors.

  Models the efficiency losses in converting DC battery voltage to 3-phase AC
  voltage for driving PMSM motors. Efficiency varies with load, typically:
  - Low load (<20%): 50-70% efficient
  - Medium load (50-70%): 90-95% efficient (peak)
  - High load (>90%): 85-90% efficient

  The inverter increases DC-side current draw to compensate for conversion
  losses: I_dc = I_ac / efficiency
  """

  efficiency_curve: list[tuple[float, float]]
  """Efficiency vs load fraction curve [(load_fraction, efficiency), ...].

  Load fraction is defined as: abs(I_motor) / I_stall

  Example for typical PMSM inverter:
      [(0.0, 0.5),   # 50% efficient at no load
       (0.2, 0.85),  # 85% at 20% load
       (0.5, 0.92),  # 92% at 50% load (peak efficiency)
       (0.7, 0.93),  # 93% at 70% load
       (1.0, 0.90)]  # 90% at full load
  """

  max_switching_frequency: float = 20000.0
  """PWM switching frequency (Hz). Typical: 16-25 kHz."""

  deadtime: float = 1e-6
  """Deadtime between high/low side switches (s). Typical: 0.5-2 μs."""

  thermal_resistance: float = 5.0
  """Thermal resistance to ambient (°C/W) for inverter module."""

  max_temperature: float = 85.0
  """Maximum junction temperature (°C) for inverter power stage."""

  def get_efficiency(
    self, load_fraction: torch.Tensor, device: str | None = None
  ) -> torch.Tensor:
    """Interpolate efficiency from load curve.

    Args:
        load_fraction: Motor load fraction [0, 1], any shape
        device: Torch device (defaults to load_fraction device)

    Returns:
        Efficiency [0, 1], same shape as load_fraction
    """
    if device is None:
      device = str(load_fraction.device)

    # Extract load and efficiency arrays from curve
    load_points = torch.tensor(
      [p[0] for p in self.efficiency_curve],
      device=device,
      dtype=torch.float32,
    )
    eff_points = torch.tensor(
      [p[1] for p in self.efficiency_curve],
      device=device,
      dtype=torch.float32,
    )

    # Clamp load fraction to curve range
    load_clamped = torch.clamp(load_fraction, min=load_points[0], max=load_points[-1])

    # Linear interpolation
    indices = torch.searchsorted(load_points, load_clamped, right=False)
    indices = torch.clamp(indices, min=1, max=len(load_points) - 1)

    load_lo = load_points[indices - 1]
    load_hi = load_points[indices]
    eff_lo = eff_points[indices - 1]
    eff_hi = eff_points[indices]

    # Interpolate: eff = eff_lo + (load - load_lo) * (eff_hi - eff_lo) / (load_hi - load_lo)
    alpha = (load_clamped - load_lo) / (load_hi - load_lo + 1e-8)
    efficiency = eff_lo + alpha * (eff_hi - eff_lo)

    # Clamp to valid range [0, 1]
    return torch.clamp(efficiency, min=0.0, max=1.0)
