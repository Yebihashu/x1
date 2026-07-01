from sim import ttl_testtools as ttl

#====================================================================
# Running tests using pytest
#====================================================================
import numpy as np
from grx import grx_math

# The GRX coords system is as follows
# ^ Y (up)            
# |              
# |              
# | . Z (front)         
# |/             
# +-------> X (right)  

def test_rotation_vector_to_rotation_matrix__roll90():      
    # DATA
    θ = 90.0 * np.pi / 180.0  # convert degrees to radians
    rotation_axis_vector = np.array([0.0, 0.0, 1.0])  # rotation around z axis
    rotation_vector = rotation_axis_vector * θ
    # EXPECTED
    # rotation vector represents a CCW θ radians around the rotaiton axis when looking towards axis tip.
    # The rotaiton axis is z, therefore the corrspnding rotation matrix is roll of 90 degrees
    rotation_matrix_expected = grx_math.parameters_to_transform(grx_math.TransformParams(x=0,y=0,z=0,roll=90,pitch=0,yaw=0))
    # TEST
    rotation_matrix_result = grx_math.rotation_vector_to_rotation_matrix(rotation_vector)
    assert np.allclose(rotation_matrix_result, rotation_matrix_expected,    rtol=1e-05, atol=1e-08)

def test_rotation_vector_to_rotation_matrix__pitch90():      
    # DATA
    θ = 90.0 * np.pi / 180.0  # convert degrees to radians
    rotation_axis_vector = np.array([1.0, 0.0, 0.0])  # rotation around x axis
    rotation_vector = rotation_axis_vector * θ
    # EXPECTED
    # rotation vector represents a CCW θ radians around the rotaiton axis when looking towards axis tip.
    # The rotaiton axis is x, therefore the corrspnding rotation matrix is pitch of 90 degrees
    rotation_matrix_expected = grx_math.parameters_to_transform(grx_math.TransformParams(x=0,y=0,z=0,roll=0,pitch=90,yaw=0))
    # TEST
    rotation_matrix_result = grx_math.rotation_vector_to_rotation_matrix(rotation_vector)
    assert np.allclose(rotation_matrix_result, rotation_matrix_expected,    rtol=1e-05, atol=1e-08)

def test_rotation_vector_to_rotation_matrix__yaw90():      
    # DATA
    θ = 90.0 * np.pi / 180.0  # convert degrees to radians
    rotation_axis_vector = np.array([0.0, 1.0, 0.0])  # rotation around y axis
    rotation_vector = rotation_axis_vector * θ
    # EXPECTED
    # rotation vector represents a CCW θ radians around the rotaiton axis when looking towards axis tip.
    # The rotaiton axis is y, therefore the corrspnding rotation matrix is yaw of 90 degrees
    rotation_matrix_expected = grx_math.parameters_to_transform(grx_math.TransformParams(x=0,y=0,z=0,roll=0,pitch=0,yaw=90))
    # TEST
    rotation_matrix_result = grx_math.rotation_vector_to_rotation_matrix(rotation_vector)
    assert np.allclose(rotation_matrix_result, rotation_matrix_expected,    rtol=1e-05, atol=1e-08)

def test_rotation_vector_to_rotation_matrix():      
    # DATA
    θ = 37.0 * np.pi / 180.0  # convert degrees to radians
    rotation_axis_vector = np.array([-1.0, 0.4, -0.6])  
    rotation_axis_vector = rotation_axis_vector / np.linalg.norm(rotation_axis_vector)
    rotation_vector = rotation_axis_vector * θ
    world_points = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [1.0, 1.0, 1.0]
    ])
    # EXPECTED
    rotated_points_expected = np.ones(world_points.shape , dtype=np.float64)
    # v∙cos(θ)+(k×v)∙sin(θ)+k(v∙k)(1-cos(θ))
    k = rotation_axis_vector
    for i in range(world_points.shape[1]):
        v = world_points[:3,i] 
        rotated_points_expected[:3,i] = v*np.cos(θ)+(np.cross(k,v))*np.sin(θ)+k*(np.dot(v,k))*(1-np.cos(θ))
    # TEST
    rotation_matrix = grx_math.rotation_vector_to_rotation_matrix(rotation_vector)
    rotated_points_results = rotation_matrix @ world_points
    assert np.allclose(rotated_points_results, rotated_points_expected,    rtol=1e-05, atol=1e-08)


def test_rotation_vector_to_rotation_matrix__no_rotation():      
    # DATA
    θ = 0.0 * np.pi / 180.0  # convert degrees to radians
    rotation_axis_vector = np.array([1.0, 0.0, 0.0])  # rotation around x axis
    rotation_vector = rotation_axis_vector * θ
    # EXPECTED
    # we expect to have identity
    rotation_matrix_expected = np.eye(4, dtype=np.float64)
    # TEST
    rotation_matrix_result = grx_math.rotation_vector_to_rotation_matrix(rotation_vector)
    assert np.allclose(rotation_matrix_result, rotation_matrix_expected,    rtol=1e-05, atol=1e-08)

def test_rotation_vector_to_rotation_matrix__180():      
    # DATA
    θ = 180.0 * np.pi / 180.0  # convert degrees to radians
    rotation_axis_vector = np.array([-1.0, 0.4, -0.6])  
    rotation_axis_vector = rotation_axis_vector / np.linalg.norm(rotation_axis_vector)
    rotation_vector = rotation_axis_vector * θ
    world_points = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [1.0, 1.0, 1.0]
    ])
    # EXPECTED
    rotated_points_expected = np.ones(world_points.shape , dtype=np.float64)
    # v∙cos(θ)+(k×v)∙sin(θ)+k(v∙k)(1-cos(θ))
    k = rotation_axis_vector
    for i in range(world_points.shape[1]):
        v = world_points[:3,i] 
        rotated_points_expected[:3,i] = v*np.cos(θ)+(np.cross(k,v))*np.sin(θ)+k*(np.dot(v,k))*(1-np.cos(θ))
    # TEST
    rotation_matrix = grx_math.rotation_vector_to_rotation_matrix(rotation_vector)
    rotated_points_results = rotation_matrix @ world_points
    assert np.allclose(rotated_points_results, rotated_points_expected,    rtol=1e-05, atol=1e-08)

#====================================================================
# Running tests maunally
#====================================================================
if __name__ == "__main__":
    # Run TestSystem tests
    ttl.run_tests_manually()