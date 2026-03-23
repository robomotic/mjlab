"""Battery specification dataclass for rechargeable lithium-based batteries.

Supports common robotics battery chemistries:
- LiPo (Lithium Polymer): 3.7V nominal, high discharge rates
- LiFePO4 (Lithium Iron Phosphate): 3.2V nominal, flat discharge curve
- Li-ion (Lithium Ion): 3.6V nominal, balanced performance
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BatterySpecification:
  """Battery specification with electrical, thermal, and chemical properties.

  This dataclass represents a complete battery pack specification including
  cell configuration (series/parallel), electrical characteristics (voltage,
  capacity, resistance), thermal properties, and discharge limits.

  All voltage values are per-cell unless otherwise noted. Pack voltages are
  computed by multiplying cell voltages by cells_series.
  """

  # === Identity ===
  battery_id: str
  """Unique identifier for the battery (e.g., 'turnigy_6s2p_5000mah')."""

  manufacturer: str
  """Manufacturer name (e.g., 'Turnigy', 'Samsung', 'A123')."""

  model: str
  """Model designation (e.g., 'Graphene 5000mAh 6S2P 65C')."""

  # === Chemistry & Configuration ===
  chemistry: str
  """Battery chemistry: 'LiPo', 'LiFePO4', or 'Li-ion'."""

  cells_series: int
  """Number of cells in series (e.g., 6 for 6S configuration)."""

  cells_parallel: int
  """Number of cells in parallel (e.g., 2 for 2P configuration)."""

  nominal_cell_voltage: float
  """Nominal voltage per cell (V): 3.7V (LiPo), 3.2V (LiFe), 3.6V (Li-ion)."""

  max_cell_voltage: float
  """Maximum voltage per cell (V): 4.2V (LiPo), 3.65V (LiFe), 4.2V (Li-ion)."""

  min_cell_voltage: float
  """Minimum voltage per cell (V): 3.0V (LiPo), 2.5V (LiFe), 2.5V (Li-ion)."""

  min_operating_voltage: float
  """Conservative cutoff voltage per cell (V) to prevent deep discharge."""

  # === Capacity & Energy ===
  capacity_ah: float
  """Capacity in amp-hours (Ah) for the parallel group."""

  # === Electrical Properties ===
  internal_resistance: float
  """Internal resistance of the pack (Ω) at nominal SOC and temperature."""

  internal_resistance_temp_coeff: float
  """Temperature coefficient of resistance (Ω/°C)."""

  # === Discharge Characteristics ===
  max_continuous_current: float
  """Maximum continuous discharge current (A)."""

  max_burst_current: float
  """Maximum burst discharge current (A) for short durations."""

  burst_duration: float
  """Maximum duration for burst current (s)."""

  c_rating_continuous: float
  """Continuous discharge C-rating (e.g., 65C means 65 * capacity_ah)."""

  # === Thermal Properties ===
  thermal_capacity: float
  """Thermal capacity (J/°C) - heat capacity of battery pack."""

  thermal_resistance: float
  """Thermal resistance to ambient (°C/W)."""

  max_temperature: float
  """Maximum safe operating temperature (°C)."""

  min_temperature: float
  """Minimum operating temperature (°C) - low-temp cutoff."""

  ambient_temperature: float
  """Initial/ambient temperature (°C)."""

  # === Optional Fields (with defaults) ===
  energy_wh: float | None = None
  """Energy capacity in watt-hours (Wh). Computed if not provided."""

  internal_resistance_soc_curve: list[tuple[float, float]] | None = None
  """SOC-dependent resistance multiplier curve [(SOC, R_multiplier), ...].
    Example: [(0.0, 2.5), (0.2, 1.5), (1.0, 1.0)] means R increases 2.5x at 0% SOC.
    """

  c_rating_burst: float | None = None
  """Burst discharge C-rating. Computed from max_burst_current if not provided."""

  ocv_curve: list[tuple[float, float]] | None = None
  """Open-circuit voltage curve [(SOC%, voltage_per_cell), ...].
    Example: [(0.0, 3.0), (0.5, 3.75), (1.0, 4.2)] for LiPo.
    """

  # === Depth of Discharge Limits ===
  min_soc: float = 0.2
  """Minimum state of charge (0.0-1.0) - typically 20% to prevent damage."""

  max_soc: float = 1.0
  """Maximum state of charge (0.0-1.0) - typically 100%."""

  # === Physical Properties (optional) ===
  mass_kg: float | None = None
  """Mass of battery pack (kg)."""

  volume_liters: float | None = None
  """Volume of battery pack (liters)."""

  # === Cell Balancing ===
  cell_balance_tolerance: float = 0.05
  """Maximum acceptable voltage imbalance between cells (V)."""

  def __post_init__(self) -> None:
    """Compute derived properties and set default curves."""
    # Compute pack voltages from cell voltages
    self.nominal_voltage = self.nominal_cell_voltage * self.cells_series
    self.max_voltage = self.max_cell_voltage * self.cells_series
    self.min_voltage = self.min_cell_voltage * self.cells_series

    # Compute energy if not provided
    if self.energy_wh is None:
      self.energy_wh = self.capacity_ah * self.nominal_voltage

    # Compute burst C-rating if not provided
    if self.c_rating_burst is None:
      self.c_rating_burst = self.max_burst_current / self.capacity_ah

    # Default OCV curve (linear approximation if not provided)
    if self.ocv_curve is None:
      self.ocv_curve = [
        (0.0, self.min_cell_voltage),
        (0.5, self.nominal_cell_voltage),
        (1.0, self.max_cell_voltage),
      ]

    # Default resistance-SOC curve (increases at low SOC)
    if self.internal_resistance_soc_curve is None:
      self.internal_resistance_soc_curve = [
        (0.0, 2.5),  # 2.5x resistance at empty
        (0.2, 1.5),  # 1.5x at 20%
        (0.5, 1.0),  # 1.0x at 50%
        (1.0, 1.0),  # 1.0x at full
      ]
