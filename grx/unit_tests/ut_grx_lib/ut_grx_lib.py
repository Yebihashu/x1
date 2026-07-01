from sim import ttl_testtools as ttl

#====================================================================
# Running tests using pytest
#====================================================================
import numpy as np
from grx import grx_lib


def _make_transform(x=0.0, y=0.0, z=0.0, axis=(0.0, 1.0, 0.0), angle_deg=0.0):
    """Build an arbitrary, invertible 4x4 transform for testing."""
    rotation = grx_lib.axis_angle_transform(list(axis), angle_deg)
    translation = grx_lib.translation_transform(x, y, z)
    return translation @ rotation


def test_scene():
    """
    create scene object
    verify it doesn't raise any errors.
    """
    scene = grx_lib.Scene()
    assert scene is not None


def test_spwan__object():
    """
    create scene, spawn cube with some transform,
    verify it doesn't raise any errors.
    """
    scene = grx_lib.Scene()
    cube = grx_lib.Cube(name="cube", size=[1.0, 2.0, 3.0])
    transform = _make_transform(x=1.0, y=2.0, z=3.0, angle_deg=30.0)
    spawned = scene.spawn_object(cube, transform=transform)
    assert spawned is cube
    assert scene.get_object("cube") is cube


def test_spwan__child_object():
    """
    create scene, spawn cube with some transform, spawn second cube as child of the first cube with some transform,
    verify it doesn't raise any errors.
    """
    scene = grx_lib.Scene()
    parent = grx_lib.Cube(name="parent")
    child = grx_lib.Cube(name="child")
    scene.spawn_object(parent, transform=_make_transform(x=1.0, angle_deg=10.0))
    scene.spawn_object(child, transform=_make_transform(z=2.0, angle_deg=20.0), parent_object=parent)
    assert child.parent is parent
    assert child in parent.children


def test_get_tranform__relative_to_scene():
    """
    create scene, spawn cube with some transform,
    get the object transform relative to scene. verify it is correct.
    """
    scene = grx_lib.Scene()
    cube = grx_lib.Cube(name="cube")
    transform = _make_transform(x=1.0, y=2.0, z=3.0, angle_deg=45.0)
    scene.spawn_object(cube, transform=transform)

    result = cube.get_transform()  # relative to scene
    assert np.allclose(result, transform)


def test_get_tranform__relative_to_object():
    """
    create scene, spawn two cubes each with different transform
    get the second object transform relative to first object. verify it is correct.
    """
    scene = grx_lib.Scene()
    cube_a = grx_lib.Cube(name="cube_a")
    cube_b = grx_lib.Cube(name="cube_b")
    transform_a = _make_transform(x=1.0, y=0.0, z=0.0, angle_deg=30.0)
    transform_b = _make_transform(x=0.0, y=5.0, z=2.0, angle_deg=-60.0)
    scene.spawn_object(cube_a, transform=transform_a)
    scene.spawn_object(cube_b, transform=transform_b)

    result = cube_b.get_transform(cube_a)
    expected = np.linalg.inv(transform_a) @ transform_b
    assert np.allclose(result, expected)

    # resolving the reference by name must give the same result
    result_by_name = cube_b.get_transform("cube_a")
    assert np.allclose(result_by_name, expected)


def test_get_tranform__relative_to_parent_object():
    """
    create scene, spawn cube with some transform, spawn second cube as child of the first cube with some transform,
    get parent transform relative to scene. and verify correct
    get parent transform relative to child object. and verify correct
    get child object transform relative to scene. and verify correct
    get child object transform relative to parent object. and verify correct
    """
    scene = grx_lib.Scene()
    parent = grx_lib.Cube(name="parent")
    child = grx_lib.Cube(name="child")

    parent_transform = _make_transform(x=2.0, y=1.0, z=0.0, angle_deg=25.0)   # relative to scene
    child_local_transform = _make_transform(x=0.0, y=0.0, z=4.0, angle_deg=-40.0)  # relative to parent

    scene.spawn_object(parent, transform=parent_transform)
    scene.spawn_object(child, transform=child_local_transform, parent_object=parent)

    parent_world = parent_transform
    child_world = parent_transform @ child_local_transform

    # parent relative to scene
    assert np.allclose(parent.get_transform(), parent_world)
    # parent relative to child
    assert np.allclose(parent.get_transform(child), np.linalg.inv(child_world) @ parent_world)
    # child relative to scene
    assert np.allclose(child.get_transform(), child_world)
    # child relative to parent
    assert np.allclose(child.get_transform(parent), child_local_transform)



#====================================================================
# Running tests maunally
#====================================================================
if __name__ == "__main__":
    # Run TestSystem tests
    ttl.run_tests_manually()
