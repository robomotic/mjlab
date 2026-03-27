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

# Remote battery repositories
REMOTE_BATTERY_REPOSITORIES = [
  "https://raw.githubusercontent.com/robomotic/mujoco-batteries/master/battery_assets"
]

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


def _extract_manufacturer(battery_id: str) -> str | None:
  """Extract manufacturer from battery ID.

  Assumes manufacturer is the first underscore-separated component.

  Examples:
    >>> _extract_manufacturer("turnigy_6s2p_5000mah")
    'turnigy'
    >>> _extract_manufacturer("unitree_g1_9ah")
    'unitree'
    >>> _extract_manufacturer("no_underscore")
    None

  Args:
    battery_id: The battery identifier (e.g., "unitree_g1_9ah")

  Returns:
    Manufacturer name (lowercase) or None if no underscore
  """
  if "_" not in battery_id:
    return None
  return battery_id.split("_")[0].lower()


def _try_remote_repositories(battery_id: str) -> BatterySpecification:
  """Attempt to load battery spec from remote repositories.

  Tries each configured repository with manufacturer subdirectory pattern,
  falling back to flat structure.

  Args:
    battery_id: The battery identifier to load

  Returns:
    BatterySpecification loaded from remote repository

  Raises:
    FileNotFoundError: If battery not found in any remote repository
  """
  manufacturer = _extract_manufacturer(battery_id)
  attempted_urls = []

  for base_url in REMOTE_BATTERY_REPOSITORIES:
    # Try with manufacturer subdirectory (preferred structure)
    if manufacturer:
      url = f"{base_url}/{manufacturer}/{battery_id}.json"
      attempted_urls.append(url)
      try:
        return _load_from_url(url)
      except Exception:
        pass

    # Try flat structure (fallback)
    url = f"{base_url}/{battery_id}.json"
    attempted_urls.append(url)
    try:
      return _load_from_url(url)
    except Exception:
      pass

  # All attempts failed
  raise FileNotFoundError(
    f"Battery '{battery_id}' not found in any remote repository. "
    f"Attempted URLs:\n" + "\n".join(f"  - {u}" for u in attempted_urls)
  )


def load_battery_spec(
  battery_id: str | None = None,
  *,
  path: str | Path | None = None,
  url: str | None = None,
  file: str | Path | None = None,
) -> BatterySpecification:
  """Load battery specification from various sources.

  When loading by ID, the system searches in this order:
  1. User directory (~/.mjlab/batteries/)
  2. Project directory (./batteries/)
  3. Environment variable ($MJLAB_BATTERY_PATH)
  4. Programmatically added paths
  5. Built-in database
  6. Remote repositories (GitHub with automatic caching)

  Downloaded batteries are cached at ~/.mjlab/cache/batteries/

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
      >>> # Load from built-in database
      >>> battery = load_battery_spec("turnigy_6s2p_5000mah")

      >>> # Load from GitHub (automatic fallback)
      >>> battery = load_battery_spec("unitree_g1_9ah")

      >>> # Load from explicit URL
      >>> battery = load_battery_spec(url="https://example.com/battery.json")

      >>> # Load from file
      >>> battery = load_battery_spec(file="/path/to/battery.json")

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
  """Load battery by ID from search paths or remote repositories.

  Search order:
  1. Local search paths (user, project, env, added, built-in)
  2. Remote repositories (GitHub with automatic caching)
  """
  # Determine search paths
  if search_path is not None:
    search_paths = [Path(search_path).expanduser().resolve()]
  else:
    search_paths = get_default_search_paths()

  # Try local paths first
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

  # Try remote repositories as fallback
  try:
    return _try_remote_repositories(battery_id)
  except Exception as remote_error:
    # Provide comprehensive error message
    raise FileNotFoundError(
      f"Battery '{battery_id}' not found in local paths or remote repositories.\n"
      f"Local paths searched:\n"
      + "\n".join(f"  - {p}" for p in search_paths)
      + f"\n{remote_error}"
    ) from remote_error


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
