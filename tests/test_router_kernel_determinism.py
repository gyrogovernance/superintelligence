import pytest
from ggg_asi_router.physics import atlas_builder
from ggg_asi_router.router.kernel import RouterKernel
from pathlib import Path


def _get_atlas_dir() -> Path:
    cfg = atlas_builder.AtlasConfiguration()
    if not all([
        (cfg.output_directory / "ontology_keys.npy").exists(),
        (cfg.output_directory / "epistemology.npy").exists(),
        (cfg.output_directory / "stage_profile.npy").exists(),
        (cfg.output_directory / "loop_defects.npy").exists(),
        (cfg.output_directory / "aperture.npy").exists(),
    ]):
        pytest.skip(
            f"Complete atlas not found at {cfg.output_directory}. "
            "Run: python -m ggg_asi_router.physics.atlas_builder complete"
        )
    return cfg.output_directory


def test_kernel_determinism() -> None:
    base_dir = _get_atlas_dir()
    payload = b"deterministic-payload"

    k1 = RouterKernel.from_directory(base_dir)
    k2 = RouterKernel.from_directory(base_dir)

    sig1 = k1.step_bytes(payload)
    sig2 = k2.step_bytes(payload)

    assert k1.state_index == k2.state_index

    assert sig1.state_index == sig2.state_index
    assert sig1.state_int_hex == sig2.state_int_hex
    assert sig1.stage_profile == sig2.stage_profile
    assert sig1.loop_defects == sig2.loop_defects
    assert sig1.aperture == sig2.aperture
    assert sig1.si == sig2.si


