"""Motor database loader with flexible path resolution."""

import hashlib
import json
import os
import urllib.request
from pathlib import Path

from mjlab.motor_database.motor_spec import MotorSpecification

# Module paths
BUILTIN_MOTORS_PATH = Path(__file__).parent / "motors"

# Global search paths (can be extended via add_motor_database_path)
_SEARCH_PATHS: list[Path] = []


def get_default_search_paths() -> list[Path]:
  """Get default motor database search paths.

  Search paths are checked in the following order:
  1. User directory (~/.mjlab/motors/)
  2. Current working directory (./motors/)
  3. Environment variable (MJLAB_MOTOR_PATH, colon-separated)
  4. Added paths (via add_motor_database_path())
  5. Built-in database (mjlab package installation)

  Returns:
      List of Path objects for motor database directories.
  """
  paths = []

  # 1. User directory
  user_dir = Path.home() / ".mjlab" / "motors"
  if user_dir.exists():
    paths.append(user_dir)

  # 2. Current working directory
  cwd_motors = Path.cwd() / "motors"
  if cwd_motors.exists():
    paths.append(cwd_motors)

  # 3. Environment variable
  if "MJLAB_MOTOR_PATH" in os.environ:
    for path_str in os.environ["MJLAB_MOTOR_PATH"].split(":"):
      path = Path(path_str).expanduser()
      if path.exists():
        paths.append(path)

  # 4. Added paths
  paths.extend(_SEARCH_PATHS)

  # 5. Built-in database
  if BUILTIN_MOTORS_PATH.exists():
    paths.append(BUILTIN_MOTORS_PATH)

  return paths


def add_motor_database_path(path: str | Path) -> None:
  """Add a motor database search path.

  Args:
      path: Directory path to add to search paths.

  Raises:
      FileNotFoundError: If the path does not exist.
  """
  path = Path(path).expanduser().resolve()
  if not path.exists():
    raise FileNotFoundError(f"Motor database path does not exist: {path}")
  if not path.is_dir():
    raise NotADirectoryError(f"Motor database path is not a directory: {path}")
  _SEARCH_PATHS.append(path)


def load_motor_spec(
  motor_id: str | None = None,
  *,
  path: str | Path | None = None,
  url: str | None = None,
  file: str | Path | None = None,
) -> MotorSpecification:
  """Load motor specification from various sources.

  Args:
      motor_id: Motor ID to search for in database paths.
      path: Explicit directory to search in.
      url: Direct URL to motor JSON file.
      file: Direct path to motor JSON file.

  Returns:
      MotorSpecification instance.

  Raises:
      ValueError: If no source is provided or multiple sources are provided.
      FileNotFoundError: If motor cannot be found.

  Examples:
      >>> # Load from database by ID
      >>> motor = load_motor_spec("unitree_7520_14")

      >>> # Load from explicit path
      >>> motor = load_motor_spec("unitree_7520_14", path="/custom/motors")

      >>> # Load from URL
      >>> motor = load_motor_spec(url="https://example.com/motor.json")

      >>> # Load from file
      >>> motor = load_motor_spec(file="/absolute/path/motor.json")
  """
  # Count number of sources provided
  sources_provided = sum(x is not None for x in [motor_id, path, url, file])

  if sources_provided == 0:
    raise ValueError("Must provide motor_id, file, or url")

  # Method 1: Direct file path
  if file is not None:
    if motor_id is not None or path is not None or url is not None:
      raise ValueError("Cannot combine 'file' with other parameters")
    return _load_from_file(Path(file))

  # Method 2: Direct URL
  if url is not None:
    if motor_id is not None or path is not None:
      raise ValueError("Cannot combine 'url' with other parameters")
    return _load_from_url(url)

  # Method 3: Search by motor_id
  if motor_id is not None:
    if url is not None or file is not None:
      raise ValueError("Cannot combine 'motor_id' with 'url' or 'file'")

    if path is not None:
      # Search in explicit path only
      return _load_from_path(motor_id, Path(path))
    else:
      # Search in all default paths
      search_paths = get_default_search_paths()
      for search_path in search_paths:
        try:
          return _load_from_path(motor_id, search_path)
        except FileNotFoundError:
          continue

      raise FileNotFoundError(
        f"Motor '{motor_id}' not found in any search path. Searched: {search_paths}"
      )

  raise ValueError("Must provide motor_id, file, or url")


def _load_from_file(file_path: Path) -> MotorSpecification:
  """Load motor spec from file.

  Args:
      file_path: Path to JSON file.

  Returns:
      MotorSpecification instance.

  Raises:
      FileNotFoundError: If file does not exist.
      json.JSONDecodeError: If file is not valid JSON.
  """
  if not file_path.exists():
    raise FileNotFoundError(f"Motor spec file not found: {file_path}")

  with open(file_path, "r") as f:
    data = json.load(f)

  return MotorSpecification(**data)


def _load_from_url(url: str) -> MotorSpecification:
  """Load motor spec from URL with caching.

  Args:
      url: URL to motor JSON file.

  Returns:
      MotorSpecification instance.

  Raises:
      urllib.error.URLError: If URL cannot be fetched.
  """
  # Setup cache directory
  cache_dir = Path.home() / ".mjlab" / "cache" / "motors"
  cache_dir.mkdir(parents=True, exist_ok=True)

  # Use URL hash as cache key
  cache_key = hashlib.md5(url.encode()).hexdigest()
  cache_file = cache_dir / f"{cache_key}.json"

  # Download if not cached
  if not cache_file.exists():
    with urllib.request.urlopen(url) as response:
      data = response.read()
    cache_file.write_bytes(data)

  return _load_from_file(cache_file)


def _load_from_path(motor_id: str, search_path: Path) -> MotorSpecification:
  """Load motor spec from search path.

  Args:
      motor_id: Motor ID to search for.
      search_path: Directory to search in.

  Returns:
      MotorSpecification instance.

  Raises:
      FileNotFoundError: If motor is not found in the path.
  """
  # Try common patterns
  patterns = [
    search_path / f"{motor_id}.json",
    search_path / motor_id / f"{motor_id}.json",
  ]

  for pattern in patterns:
    if pattern.exists():
      return _load_from_file(pattern)

  # Try recursive glob pattern
  glob_matches = list(search_path.glob(f"**/{motor_id}.json"))
  if glob_matches:
    return _load_from_file(glob_matches[0])

  raise FileNotFoundError(f"Motor '{motor_id}' not found in {search_path}")
