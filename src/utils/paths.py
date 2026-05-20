from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def project_path(*parts):
    """Build a path relative to the repository root."""
    return PROJECT_ROOT.joinpath(*parts)


def data_path(filename):
    """Build a path inside the repository data directory."""
    return DATA_DIR / filename


def results_path(*parts):
    """Build a path inside the repository results directory."""
    path = RESULTS_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(path):
    """Create the parent directory for an output path."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
