"""Motor specification dataclass with electrical/thermal/mechanical properties."""

from dataclasses import dataclass
from typing import Any


@dataclass
class MotorSpecification:
  """Motor specification with electrical, thermal, and mechanical properties.

  This class defines a comprehensive motor specification including identity,
  mechanical properties, electrical characteristics, performance metrics,
  thermal behavior, and feedback/control parameters.

  Fields are categorized by requirement level:

  **Required Fields (no defaults):**
  These fields are physics-critical and must be specified for each motor:
      motor_id: Unique identifier for the motor (e.g., "unitree_7520_14").
      manufacturer: Motor manufacturer name.
      model: Motor model number.
      voltage_range: (min, max) operating voltage range in volts.
      resistance: Winding resistance in ohms.
      inductance: Winding inductance in henries.
      motor_constant_kt: Torque constant in N⋅m/A.
      motor_constant_ke: Back-EMF constant in V⋅s/rad.
      peak_torque: Maximum instantaneous torque in N⋅m.
      no_load_speed: No-load speed at zero torque in rad/s.
      thermal_resistance: Thermal resistance (junction to ambient) in °C/W.
      thermal_time_constant: Thermal time constant in seconds.
      max_winding_temperature: Maximum winding temperature in °C.

  **Optional Fields (with sensible defaults):**
      step_file: Optional path to STEP CAD file.
      stl_file: Optional path to STL mesh file.
      gear_ratio: Gear reduction ratio (default: 1.0, direct drive).
      reflected_inertia: Reflected rotor inertia after gearbox in kg⋅m² (default: 0.0).
      rotation_angle_range: (min, max) joint angle range in radians (default: ±π).
      weight: Motor weight in kilograms (default: 0.0).
          For calculating robot mass distribution.
          Example: Maxon EC-i 40: 0.390 kg, Faulhaber 2657: 0.140 kg
      friction_static: Static friction torque in N⋅m (Coulomb friction, default: 0.0).
          Applied at zero velocity for stick-slip modeling.
      friction_dynamic: Dynamic friction coefficient in N⋅m⋅s/rad (viscous damping, default: 0.0).
          Velocity-proportional friction distinct from PD controller damping.
      stall_torque: Stall torque at zero speed in N⋅m (default: 10.0).
      continuous_torque: Continuous rated torque in N⋅m (default: 10.0).
      no_load_current: No-load current draw in amperes (default: 0.5).
      stall_current: Stall current at zero speed in amperes (default: 10.0).
      operating_current: Nominal operating current in amperes (default: 3.0).
      ambient_temperature: Ambient temperature in °C (default: 25.0, room temp).
      number_of_pole_pairs: Number of magnetic pole pairs (default: None).
          Used for commutation frequency calculation: f_comm = (pole_pairs × speed) / (2π)
          Example: 7 for Maxon EC-i motors, 2 for Faulhaber motors
      commutation: Sensor type for commutation - "Hall", "Encoder", "Sensorless", or None (default: None).
          Metadata for control system compatibility.
      max_speed: Maximum mechanical speed in rad/s (bearing/mechanical limit, default: None).
          Different from no_load_speed (electrical limit at rated voltage).
          Example: 8000 RPM = 837.7 rad/s
      encoder_resolution: Encoder resolution in counts/revolution (default: 2048).
      encoder_type: Encoder type ("incremental" or "absolute", default: "incremental").
      feedback_sensors: List of available feedback sensors (default: ["position", "velocity"]).
      protocol: Communication protocol ("PWM", "CAN", "UART", etc., default: "PWM").
      protocol_params: Protocol-specific configuration parameters (default: {}).

  **Physics-Critical Fields:**
  The following fields are directly used in the electrical motor actuator physics:
  - RL circuit dynamics: resistance, inductance, motor_constant_kt, motor_constant_ke
  - Voltage limiting: voltage_range
  - Thermal dynamics: thermal_resistance, thermal_time_constant, max_winding_temperature, ambient_temperature
  - DC motor saturation: peak_torque, no_load_speed

  **Future Physics Extensions:**
  The following optional fields are available but not currently used in physics simulations:
  - number_of_pole_pairs: Reserved for BLDC/PMSM commutation frequency modeling
  - max_speed: Reserved for mechanical speed limit enforcement beyond electrical no_load_speed
  - weight: Reserved for total actuator mass calculations
  - friction_static: Reserved for Coulomb friction in torque output
  - friction_dynamic: Reserved for velocity-proportional friction
  - commutation: Metadata for control system compatibility
  """

  # Identity (required)
  motor_id: str
  manufacturer: str
  model: str

  # Electrical Properties (required - motor-specific)
  voltage_range: tuple[float, float]
  resistance: float
  inductance: float
  motor_constant_kt: float
  motor_constant_ke: float

  # Performance Characteristics (required - motor-specific)
  peak_torque: float
  no_load_speed: float

  # Thermal Properties (required - motor-specific)
  thermal_resistance: float
  thermal_time_constant: float
  max_winding_temperature: float

  # Electrical Properties (optional - additional characteristics)
  number_of_pole_pairs: int | None = None
  commutation: str | None = None

  # Performance Characteristics (optional - additional limits)
  max_speed: float | None = None

  # 3D Assets (optional)
  step_file: str | None = None
  stl_file: str | None = None

  # Mechanical Properties (optional - with defaults)
  gear_ratio: float = 1.0
  reflected_inertia: float = 0.0
  rotation_angle_range: tuple[float, float] = (-3.14159, 3.14159)
  weight: float = 0.0
  friction_static: float = 0.0
  friction_dynamic: float = 0.0

  # Performance Characteristics (optional - with defaults)
  stall_torque: float = 10.0
  continuous_torque: float = 10.0
  no_load_current: float = 0.5
  stall_current: float = 10.0
  operating_current: float = 3.0

  # Thermal Properties (optional - with defaults)
  ambient_temperature: float = 25.0

  # Feedback & Control (optional - with defaults)
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
