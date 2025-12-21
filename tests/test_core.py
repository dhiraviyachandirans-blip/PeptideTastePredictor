from peptide_core import compute_features, compute_peptide_properties, validate_sequence, compute_docking_score


def test_validate_sequence_ok():
    ok, s = validate_sequence("ACDEFGHIK")
    assert ok
    assert s == "ACDEFGHIK"


def test_validate_sequence_invalid_chars():
    ok, reason = validate_sequence("AXZ")
    assert not ok
    assert "Invalid characters" in reason


def test_compute_peptide_properties():
    props = compute_peptide_properties("ACDEFGHIK")
    assert "Molecular Weight" in props
    assert isinstance(props["Molecular Weight"], float)


def test_compute_docking_score():
    # confidence 1.0 and high gravy should give a high score
    s1 = compute_docking_score(1.5, 1.0)
    assert isinstance(s1, int)
    assert 80 <= s1 <= 100

    # low confidence, low gravy -> low score
    s2 = compute_docking_score(-1.5, 0.05)
    assert 0 <= s2 <= 30

    # edge cases: non-numeric input handled gracefully
    s3 = compute_docking_score('nan', None)
    assert isinstance(s3, int)
    assert 0 <= s3 <= 100


def test_compute_features_minimal():
    df = compute_features(["ACDEFGHIK"])
    assert not df.empty
    assert df.shape[0] == 1
