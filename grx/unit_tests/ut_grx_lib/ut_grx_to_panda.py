from sim import ttl_testtools as ttl

#====================================================================
# Running tests using pytest
#====================================================================
import numpy as np
import pytest
from grx import grx_lib


def test_grx_to_panda_conversion():
    panda3d = pytest.importorskip("panda3d.core")

    grx_transform = (
        grx_lib.translation_transform(x=1.1, y=-2.2, z=3.3)
        @ grx_lib.axis_angle_transform(axis=[-1.0, 0.4, -0.6], angle_deg=37.0)
    )

    panda_matrix = grx_lib._grx_to_panda_mat4(grx_transform)
    panda_row_major = np.array(
        [[panda_matrix.getCell(r, c) for c in range(4)] for r in range(4)],
        dtype=np.float64,
    )
    expected_panda_row_major = (
        grx_lib._GRX_PANDA_BASIS @ grx_transform @ grx_lib._GRX_PANDA_BASIS
    ).T

    assert panda_row_major.shape == (4, 4)
    assert np.allclose(panda_row_major, expected_panda_row_major, rtol=1e-5, atol=1e-8)


def test_panda_to_grx_conversion():
    grx_transform = (
        grx_lib.translation_transform(x=-3.5, y=0.25, z=8.0)
        @ grx_lib.axis_angle_transform(axis=[0.2, 1.0, -0.3], angle_deg=-55.0)
    )

    panda_row_major = (
        grx_lib._GRX_PANDA_BASIS @ grx_transform @ grx_lib._GRX_PANDA_BASIS
    ).T

    grx_result = grx_lib._panda_mat4_to_grx(panda_row_major)

    assert grx_result.shape == (4, 4)
    assert np.allclose(grx_result, grx_transform, rtol=1e-5, atol=1e-8)


#====================================================================
# Running tests maunally
#====================================================================
if __name__ == "__main__":
    # Run TestSystem tests
    ttl.run_tests_manually()
