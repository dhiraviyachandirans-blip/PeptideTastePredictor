import shutil
import subprocess
import re
from typing import List, Tuple, Optional
import tempfile
import os


def detect_vina() -> Optional[str]:
    """Return path to a docking binary (vina or smina) if available, otherwise None."""
    for name in ("vina", "smina"):
        path = shutil.which(name)
        if path:
            return path
    return None


def parse_vina_log(log_text: str) -> List[float]:
    """Parse affinity scores from a Vina log text.

    Returns a list of affinities (kcal/mol) from the log in the order they appear.
    The parser looks for common lines such as "REMARK VINA RESULT: -7.8" or "Affinity: -7.8".
    """
    scores = []
    # Search for "REMARK VINA RESULT" lines
    for m in re.finditer(r"REMARK\s+VINA\s+RESULT:\s*([\-\d\.]+)", log_text, re.IGNORECASE):
        try:
            scores.append(float(m.group(1)))
        except Exception:
            continue

    # Also look for 'Affinity: -7.8' patterns
    for m in re.finditer(r"Affinity:\s*([\-\d\.]+)", log_text, re.IGNORECASE):
        try:
            scores.append(float(m.group(1)))
        except Exception:
            continue

    return scores


def compute_pdb_center(pdb_text: str) -> Tuple[float, float, float]:
    """Compute simple geometric center from ATOM/HETATM coordinates in a PDB string.

    Returns (x, y, z). If parsing fails, returns (0.0, 0.0, 0.0).
    """
    xs, ys, zs = [], [], []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                xs.append(x); ys.append(y); zs.append(z)
            except Exception:
                continue
    if not xs:
        return 0.0, 0.0, 0.0
    return (sum(xs)/len(xs), sum(ys)/len(ys), sum(zs)/len(zs))


def run_vina(vina_path: str, receptor_pdb_path: str, ligand_pdb_path: str, center: Tuple[float,float,float], size: Tuple[float,float,float], out_dir: Optional[str]=None, exhaustiveness: int = 8, num_modes: int = 9) -> Tuple[List[float], str]:
    """Run Vina (or smina) with provided inputs and return (scores, log_text).

    - vina_path: path to executable
    - receptor_pdb_path, ligand_pdb_path: file paths
    - center: (x,y,z) center of the search box
    - size: (x,y,z) size of the search box (in Angstroms)
    - out_dir: directory to write outputs (if None a temp dir is used)

    Returns: (scores list, stdout+stderr log as string)

    Note: this function writes temporary files and calls the executable. It does not
    attempt to convert formats (assumes receptor/ligand are already in suitable formats).
    """
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="vina_out_")
    out_pdbqt = os.path.join(out_dir, "out.pdbqt")
    log_file = os.path.join(out_dir, "vina.log")

    # Build command: prefer smina compatible args if 'smina' in name
    base = os.path.basename(vina_path).lower()
    cmd = [vina_path,
           "--receptor", receptor_pdb_path,
           "--ligand", ligand_pdb_path,
           "--out", out_pdbqt,
           "--center_x", str(center[0]), "--center_y", str(center[1]), "--center_z", str(center[2]),
           "--size_x", str(size[0]), "--size_y", str(size[1]), "--size_z", str(size[2]),
           "--exhaustiveness", str(exhaustiveness), "--num_modes", str(num_modes)]

    try:
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        log_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    except Exception as e:
        raise RuntimeError(f"Failed to run docking binary: {e}")

    # Parse scores
    scores = parse_vina_log(log_text)
    return scores, log_text
