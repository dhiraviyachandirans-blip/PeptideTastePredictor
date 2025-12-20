from docking_utils import parse_vina_log, compute_pdb_center


def test_parse_vina_log_variants():
    log1 = """
REMARK VINA RESULT: -7.8 1
REMARK VINA RESULT: -6.2 2
"""
    scores = parse_vina_log(log1)
    assert -7.8 in scores and -6.2 in scores

    log2 = """
Affinity: -9.4 (kcal/mol)
Affinity: -8.1 (kcal/mol)
"""
    scores2 = parse_vina_log(log2)
    assert -9.4 in scores2 and -8.1 in scores2


def test_compute_pdb_center():
    # Minimal PDB with two atoms
    pdb = """
ATOM      1  N   ALA A   1      11.104  13.207  -2.100  1.00  0.00           N
ATOM      2  CA  ALA A   1      12.000  14.000  -1.000  1.00  0.00           C
"""
    cx, cy, cz = compute_pdb_center(pdb)
    assert round(cx,3) == round((11.104+12.000)/2,3)
    assert round(cy,3) == round((13.207+14.000)/2,3)
    assert round(cz,3) == round((-2.1 + -1.0)/2,3)
