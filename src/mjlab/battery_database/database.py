"""Battery database loader with flexible path resolution.

Follows the same pattern as motor_database for consistency.
"""

import hashlib
import json
import os
import urllib.request
from pathlib import Path

from mjlab.battery_database.battery_spec import BatterySpecification

# Module paths
BUILTIN_BATTERIES_PATH = Path(__file__).parent / "batteries"

# Global search paths (can be extended via add_battery_database_path)
_SEARCH_PATHS: list[Path] = []


def get_default_search_paths() -> list[Path]:
  """Get default battery database search paths.

  Search paths are checked in the following order:
  1. User directory (~/.mjlab/batteries/)
  2. Current working directory (./batteries/)
  3. Environment variable (MJLAB_BATTERY_PATH, colon-separated)
  4. Added paths (via add_battery_database_path())
  5. Built-in database (mjlab package installation)

  Returns:
      List of Path objects for battery database directories.
  """
  paths = []

  # 1. User directory
  user_dir = Path.home() / ".mjlab" / "batteries"
  if user_dir.exists():
    paths.append(user_dir)

  # 2. Current working directory
  cwd_batteries = Path.cwd() / "batteries"
  if cwd_batteries.exists():
    paths.append(cwd_batteries)

  # 3. Environment variable
  if "MJLAB_BATTERY_PATH" in os.environ:
    for path_str in os.environ["MJLAB_BATTERY_PATH"].split(":"):
      path = Path(path_str).expanduser()
      if path.exists():
        paths.append(path)

  # 4. Added paths
  paths.extend(_SEARCH_PATHS)

  # 5. Built-in database
  if BUILTIN_BATTERIES_PATH.exists():
    paths.append(BUILTIN_BATTERIES_PATH)

  return paths


def add_battery_database_path(path: str | Path) -> None:
  """Add a battery database search path.

  Args:
      path: Directory path to add to search paths.

  Raises:
      FileNotFoundError: If the path does not exist.
  """
  path = Path(path).expanduser().resolve()
  if not path.exists():
    raise FileNotFoundError(f"Battery database path does not exist: {path}")
  if not path.is_dir():
    raise NotADirectoryError(f"Battery database path is not a directory: {path}")
  _SEARCH_PATHS.append(path)


def load_battery_spec(
  battery_id: str | None = None,
  *,
  path: str | Path | None = None,
  url: str | None = None,
  file: str | Path | None = None,
) -> BatterySpecification:
  """Load battery specification from various sources.

  Args:
      battery_id: Battery ID to search for in database paths.
      path: Explicit directory to search in.
      url: Direct URL to battery JSON file.
      file: Direct path to battery JSON file.

  Returns:
      BatterySpecification instance.

  Raises:
      ValueError: If no source is provided or multiple sources are provided.
      FileNotFoundError: If battery cannot be found.

  Examples:
      >>> # Load from database by ID
      >>> battery = load_battery_spec("turnigy_6s2p_5000mah")

      >>> # Load from file
      >>> battery = load_battery_spec(file="/path/to/battery.json")

      >>> # Load from URL
      >>> battery = load_battery_spec(url="https://example.com/battery.json")

      >>> # Search in specific directory
      >>> battery = load_battery_spec("custom_battery", path="/my/batteries")
  """
  # Validate arguments
  sources = sum([battery_id is not None, url is not None, file is not None])
  if sources == 0:
    raise ValueError("Must provide one of: battery_id, url, or file")
  if sources > 1:
    raise ValueError("Can only provide one of: battery_id, url, or file")

  # Load from URL
  if url is not None:
    return _load_from_url(url)

  # Load from file
  if file is not None:
    file_path = Path(file).expanduser().resolve()
    return _load_from_file(file_path)

  # Load from battery_id
  if battery_id is not None:
    return _load_by_id(battery_id, path)

  raise ValueError("Unreachable")  # Should never get here


def _load_by_id(
  battery_id: str, search_path: str | Path | None = None
) -> BatterySpecification:
  """Load battery by ID from search paths."""
  # Determine search paths
  if search_path is not None:
    search_paths = [Path(search_path).expanduser().resolve()]
  else:
    search_paths = get_default_search_paths()

  # Search for battery_id.json in search paths
  for base_path in search_paths:
    # Try direct file: battery_id.json
    candidate = base_path / f"{battery_id}.json"
    if candidate.exists():
      return _load_from_file(candidate)

    # Try subdirectory: battery_id/battery_id.json
    candidate = base_path / battery_id / f"{battery_id}.json"
    if candidate.exists():
      return _load_from_file(candidate)

    # Try recursive glob search
    for candidate in base_path.rglob(f"{battery_id}.json"):
      return _load_from_file(candidate)

  # Not found
  raise FileNotFoundError(
    f"Battery '{battery_id}' not found in search paths: "
    f"{[str(p) for p in search_paths]}"
  )


def _load_from_file(file_path: Path) -> BatterySpecification:
  """Load battery specification from JSON file."""
  if not file_path.exists():
    raise FileNotFoundError(f"Battery file not found: {file_path}")

  with open(file_path, "r") as f:
    data = json.load(f)

  return BatterySpecification(**data)


def _load_from_url(url: str) -> BatterySpecification:
  """Load battery specification from URL with caching."""
  # Compute cache path
  cache_dir = Path.home() / ".mjlab" / "cache" / "batteries"
  cache_dir.mkdir(parents=True, exist_ok=True)

  # Use MD5 hash of URL as cache filename
  url_hash = hashlib.md5(url.encode()).hexdigest()
  cache_file = cache_dir / f"{url_hash}.json"

  # Check cache
  if cache_file.exists():
    return _load_from_file(cache_file)

  # Download from URL
  try:
    with urllib.request.urlopen(url) as response:
      data = json.loads(response.read().decode())
  except Exception as e:
    raise FileNotFoundError(f"Failed to load battery from URL {url}: {e}") from e

  # Cache to disk
  with open(cache_file, "w") as f:
    json.dump(data, f, indent=2)

  return BatterySpecification(**data)
