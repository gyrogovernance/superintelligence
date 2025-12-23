import numpy as np

from ggg_asi_router.physics import governance


def test_gene_mac_roundtrip() -> None:
    tensor = governance.GENE_Mac_S.copy()
    state_int = governance.tensor_to_int(tensor)
    tensor_rt = governance.int_to_tensor(state_int)
    assert np.array_equal(tensor, tensor_rt)


def test_byte_to_action_definition() -> None:
    for b in range(256):
        assert governance.byte_to_action(b) == ((b ^ 0xAA) & 0xFF)


