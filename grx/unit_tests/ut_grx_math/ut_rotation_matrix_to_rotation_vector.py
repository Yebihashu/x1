from sim import ttl_testtools as ttl

#====================================================================
# Running tests using pytest
#====================================================================
import numpy as np
from grx import grx_math

# The GRX coords system is as follows,
# ^ Y (up)            
# |              
# |              
# | . Z (front)         
# |/             
# +-------> X (right)  


def test_rotation_matrix_to_rotation_vector__roll90():
    # DATA
    θ = 90.0 * np.pi / 180.0  # convert degrees to radians
    rotation_axis_vector = np.array([0.0, 0.0, 1.0])  # rotation around z axis
    # EXPECTED
    rotation_vector_expected = rotation_axis_vector * θ
    # TEST
    rotation_matrix = grx_math.rotation_vector_to_rotation_matrix(rotation_vector_expected)
    rotation_vector_result = grx_math.rotation_matrix_to_rotation_vector(rotation_matrix)
    assert np.allclose(rotation_vector_result, rotation_vector_expected,    rtol=1e-05, atol=1e-08)


def test_rotation_matrix_to_rotation_vector__pitch90():      
    # DATA
    θ = 90.0 * np.pi / 180.0  # convert degrees to radians
    rotation_axis_vector = np.array([1.0, 0.0, 0.0])  # rotation around x axis
    # EXPECTED
    rotation_vector_expected = rotation_axis_vector * θ
    # TEST
    rotation_matrix = grx_math.rotation_vector_to_rotation_matrix(rotation_vector_expected)
    rotation_vector_result = grx_math.rotation_matrix_to_rotation_vector(rotation_matrix)
    assert np.allclose(rotation_vector_result, rotation_vector_expected,    rtol=1e-05, atol=1e-08)

def test_rotation_matrix_to_rotation_vector__yaw90():      
    # DATA
    θ = 90.0 * np.pi / 180.0  # convert degrees to radians
    rotation_axis_vector = np.array([0.0, 1.0, 0.0])  # rotation around y axis
    # EXPECTED
    rotation_vector_expected = rotation_axis_vector * θ
    # TEST
    rotation_matrix = grx_math.rotation_vector_to_rotation_matrix(rotation_vector_expected)
    rotation_vector_result = grx_math.rotation_matrix_to_rotation_vector(rotation_matrix)
    assert np.allclose(rotation_vector_result, rotation_vector_expected,    rtol=1e-05, atol=1e-08)

def test_rotation_matrix_to_rotation_vector__minus_yaw90():      
    # DATA
    θ = -90.0 * np.pi / 180.0  # convert degrees to radians
    rotation_axis_vector = np.array([0.0, 1.0, 0.0])  # rotation around y axis
    # EXPECTED
    rotation_vector_expected = rotation_axis_vector * θ
    # TEST
    rotation_matrix = grx_math.rotation_vector_to_rotation_matrix(rotation_vector_expected)
    rotation_vector_result = grx_math.rotation_matrix_to_rotation_vector(rotation_matrix)
    assert np.allclose(rotation_vector_result, rotation_vector_expected,    rtol=1e-05, atol=1e-08)

def test_rotation_matrix_to_rotation_vector():      
    # DATA
    θ = 37.0 * np.pi / 180.0  # convert degrees to radians
    rotation_axis_vector = np.array([-1.0, 0.4, -0.6])  
    rotation_axis_vector = rotation_axis_vector / np.linalg.norm(rotation_axis_vector)
    # EXPECTED
    rotation_vector_expected = rotation_axis_vector * θ
    # TEST
    rotation_matrix = grx_math.rotation_vector_to_rotation_matrix(rotation_vector_expected)
    rotation_vector_result = grx_math.rotation_matrix_to_rotation_vector(rotation_matrix)
    assert np.allclose(rotation_vector_result, rotation_vector_expected,    rtol=1e-05, atol=1e-08)



def test_rotation_matrix_to_rotation_vector__no_rotation():      
    # DATA
    θ = 0.0 * np.pi / 180.0  # convert degrees to radians
    rotation_axis_vector = np.array([1.0, 0.0, 0.0])  # rotation around x axis
    # EXPECTED
    rotation_vector_expected = rotation_axis_vector * θ
    # TEST
    rotation_matrix = grx_math.rotation_vector_to_rotation_matrix(rotation_vector_expected)
    rotation_vector_result = grx_math.rotation_matrix_to_rotation_vector(rotation_matrix)
    assert np.allclose(rotation_vector_result, rotation_vector_expected,    rtol=1e-05, atol=1e-08)

def test_rotation_matrix_to_rotation_vector__180():      
    # DATA
    θ = 180.0 * np.pi / 180.0  # convert degrees to radians
    rotation_axis_vector = np.array([-1.0, 0.4, -0.6])  
    rotation_axis_vector = rotation_axis_vector / np.linalg.norm(rotation_axis_vector)
    # EXPECTED
    rotation_vector = rotation_axis_vector * θ
    world_points = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [1.0, 1.0, 1.0]
    ])
    # EXPECTED
    rotated_points_expected = np.ones(world_points.shape , dtype=np.float64)
    # v∙cos(θ)+(v×k)∙sin(θ)+k(v∙k)(1-cos(θ))
    k = rotation_axis_vector
    for i in range(world_points.shape[1]):
        v = world_points[:3,i] 
        rotated_points_expected[:3,i] = v*np.cos(θ)+(np.cross(v,k))*np.sin(θ)+k*(np.dot(v,k))*(1-np.cos(θ))

    # TEST
    rotation_matrix = grx_math.rotation_vector_to_rotation_matrix(rotation_vector)
    rotation_vector_result = grx_math.rotation_matrix_to_rotation_vector(rotation_matrix)

    # for angle of 180 , the direction of rotation axis may not be preserved, and it doesn't matter because the rotation is the same
    # Thus, the correct test is by checking actuall rotation  
    rotated_points_results = np.ones(world_points.shape , dtype=np.float64)
    # v∙cos(θ)+(v×k)∙sin(θ)+k(v∙k)(1-cos(θ))
    θ = np.linalg.norm(rotation_vector_result)
    k = rotation_vector_result / θ
    for i in range(world_points.shape[1]):
        v = world_points[:3,i] 
        rotated_points_results[:3,i] = v*np.cos(θ)+(np.cross(v,k))*np.sin(θ)+k*(np.dot(v,k))*(1-np.cos(θ))
     
        
    assert np.allclose(rotated_points_results, rotated_points_expected,    rtol=1e-05, atol=1e-08)

#====================================================================
# Running tests maunally
#====================================================================
if __name__ == "__main__":
    # Run TestSystem tests
    ttl.run_tests_manually()