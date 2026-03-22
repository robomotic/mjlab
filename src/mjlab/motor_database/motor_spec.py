"""Motor specification dataclass with electrical/thermal/mechanical properties."""

from dataclasses import dataclass
from typing import Any


@dataclass
class MotorSpecification:
  """Motor specification with electrical, thermal, and mechanical properties.

  This class defines a comprehensive motor specification including identity,
  mechanical properties, electrical characteristics, performance metrics,
  thermal behavior, and feedback/control parameters.

  Attributes:
      motor_id: Unique identifier for the motor (e.g., "unitree_7520_14").
      manufacturer: Motor manufacturer name.
      model: Motor model number.
      step_file: Optional path to STEP CAD file.
      stl_file: Optional path to STL mesh file.
      gear_ratio: Gear reduction ratio.
      reflected_inertia: Reflected rotor inertia after gearbox (kg⋅m²).
      rotation_angle_range: (min, max) joint angle range in radians.
      voltage_range: (min, max) operating voltage range in volts.
      resistance: Winding resistance in ohms.
      inductance: Winding inductance in henries.
      motor_constant_kt: Torque constant in N⋅m/A.
      motor_constant_ke: Back-EMF constant in V⋅s/rad.
      stall_torque: Stall torque at zero speed in N⋅m.
      peak_torque: Maximum instantaneous torque in N⋅m.
      continuous_torque: Continuous rated torque in N⋅m.
      no_load_speed: No-load speed at zero torque in rad/s.
      no_load_current: No-load current draw in amperes.
      stall_current: Stall current at zero speed in amperes.
      operating_current: Nominal operating current in amperes.
      thermal_resistance: Thermal resistance (junction to ambient) in °C/W.
      thermal_time_constant: Thermal time constant in seconds.
      max_winding_temperature: Maximum winding temperature in °C.
      ambient_temperature: Ambient temperature in °C.
      encoder_resolution: Encoder resolution in counts/revolution.
      encoder_type: Encoder type ("incremental" or "absolute").
      feedback_sensors: List of available feedback sensors.
      protocol: Communication protocol ("PWM", "CAN", "UART", etc.).
      protocol_params: Protocol-specific configuration parameters.
  """

  # Identity
  motor_id: str
  manufacturer: str
  model: str

  # 3D Assets (optional)
  step_file: str | None = None
  stl_file: str | None = None

  # Mechanical Properties
  gear_ratio: float = 1.0
  reflected_inertia: float = 0.0
  rotation_angle_range: tuple[float, float] = (-3.14159, 3.14159)

  # Electrical Properties
  voltage_range: tuple[float, float] = (0.0, 24.0)
  resistance: float = 1.0
  inductance: float = 0.001
  motor_constant_kt: float = 0.1
  motor_constant_ke: float = 0.1

  # Performance Characteristics
  stall_torque: float = 10.0
  peak_torque: float = 10.0
  continuous_torque: float = 10.0
  no_load_speed: float = 10.0
  no_load_current: float = 0.5
  stall_current: float = 10.0
  operating_current: float = 3.0

  # Thermal Properties
  thermal_resistance: float = 5.0
  thermal_time_constant: float = 300.0
  max_winding_temperature: float = 120.0
  ambient_temperature: float = 25.0

  # Feedback & Control
  encoder_resolution: int = 2048
  encoder_type: str = "incremental"
  feedback_sensors: list[str] | None = None
  protocol: str = "PWM"
  protocol_params: dict[str, Any] | None = None

  def __post_init__(self) -> None:
    """Initialize default values for mutable fields."""
    if self.feedback_sensors is None:
      self.feedback_sensors = ["position", "velocity"]
    if self.protocol_params is None:
      self.protocol_params = {}
