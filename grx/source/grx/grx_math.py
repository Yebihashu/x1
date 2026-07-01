from numbers import Real
from typing import List
from typing import Tuple
import math 
import numpy as np
from scipy.spatial.transform import Rotation

"""
design:
Coordinates system in grx is according to "\\grx\\source\\grx\\docs\\coordinates system.md"
Units are meters and degrees.
Transform is 4x4 matrix in column major order.
this means, transforming a homogeniouse vector point is done by transform_matrix @ vector_point


### Axes directions:
x - right
y - up
z - front

### Rotation directions:
roll  - around z - positive is counter clock wise
pitch - around x - positive is down
yaw   - around y - positive right 

### Order of rotations:
1. roll
2. pitch
3. yaw

"""


CLOSE_TO_ZERO = 0.000001
TMP_A = np.array([[0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0]], dtype=np.float64 )                   
TMP_At = TMP_A.transpose()


class Point():    
    """
    point structure
    """
    def __init__(self, x=0.0,y=0.0):
        self.x = x
        self.y = y
    
    def __str__(self):
        a = 20
        return "(x="+str(round(self.x,a))+", y="+str(round(self.y,a))
    
    def __eq__(self, other): 
        return (self.x == other.x) and (self.y == other.y)


class Rect():    
    """
    rect structure
    """
    def __init__(self, x=0.0,y=0.0,w=0.0,h=0.0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    
    def __str__(self):
        a = 20
        return "(x="+str(round(self.x,a))+", y="+str(round(self.y,a))+", w="+str(round(self.w,a))+", h="+str(round(self.h,a))+")"
    
    def __eq__(self, other): 
        return (self.x == other.x) and (self.y == other.y) and (self.w == other.w) and (self.h == other.h)


class TransformParams():
    def __init__(self, x=0.0,y=0.0,z=0.0,roll=0.0,pitch=0.0,yaw=0.0, sx=1.0, sy=1.0, sz=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
        self.sx = sx
        self.sy = sy
        self.sz = sz
    def __str__(self):
        return f"x={self.x}\ny={self.y}\nz={self.z}\nroll={self.roll}\npitch={self.pitch}\nyaw={self.yaw}\nsx={self.sx}\nsy={self.sy}\nsz={self.sz}"
    
    def __eq__(self, other, accuracy=CLOSE_TO_ZERO):
        if not isinstance(other, TransformParams):
            return False
        return (abs(self.x - other.x) < accuracy and
                abs(self.y - other.y) < accuracy and
                abs(self.z - other.z) < accuracy and
                abs(self.roll - other.roll) < accuracy and
                abs(self.pitch - other.pitch) < accuracy and
                abs(self.yaw - other.yaw) < accuracy and
                abs(self.sx - other.sx) < accuracy and
                abs(self.sy - other.sy) < accuracy and
                abs(self.sz - other.sz) < accuracy)



def clamp_val(x : Real, min_val: Real, max_val: Real) -> Real:
    """
    clamp value to a range
    Args:
        x        :  input value
        min_val  :  minimal range vlaue
        max_val  :  maximal range vlaue
    Returns:
        x       - if min_val <= x <= max_val
        min_val - if min_val > x
        max_val - if max_val < x
    """
    return max(min_val, min(x, max_val))


def parameters_to_transform(params : TransformParams) -> np.array:  
    """
    Create transform from parameters of translation, rotaiton [degs] and scaling
    See axes system diagram here above
    Args:
        params : transform parameters  : units [meters] , [degs]
   Returns:
        transform matrix - 2d numpy homogenoius transform
   """                
     # For carla, rotations order is (1)roll (2)pitch (3)yaw.
     # The axies and rotation directions are as shown in: https://confluence.mobileye.com/display/SG/Coordinates+systems+and+conversions+for+ME+and+Carla
     # The rotation&scale matrix is therefore:
     # (c and s stands for cos and sin, r,p and y stands for roll,pitch and yaw. sx/sy/sz are the scale factors)
     #     |(cy*cp) * sx, (cy*sp*sr-sy*cr) * sy, (-cy*sp*cr-sy*sr) * sz, 0.0|
     # m = |(sy*cp) * sx, (sy*sp*sr+cy*cr) * sy, (-sy*sp*cr+cy*sr) * sz, 0.0|
     #     |sp * sx,      (-cp*sr) * sy,         (cp*cr) * sz,           0.0|        
     #     |0.0,          0.0,                   0.0,                    1.0| 
     #    
    r = -params.roll*math.pi/180.0
    p = -params.pitch*math.pi/180.0
    w = params.yaw*math.pi/180.0
    cr = math.cos(r)
    sr = math.sin(r)
    cp = math.cos(p)
    sp = math.sin(p)
    cw = math.cos(w)
    sw = math.sin(w)

    sx = params.sx
    sy = params.sy
    sz = params.sz

    tx = params.x
    ty = params.y
    tz = params.z

    m = np.array([  [(cw*cp)*sx, (cw*sp*sr-sw*cr)*sy, (-cw*sp*cr-sw*sr)*sz, tx],
                    [(sw*cp)*sx, (sw*sp*sr+cw*cr)*sy, (-sw*sp*cr+cw*sr)*sz, ty],
                    [sp*sx,      (-cp*sr)*sy,         (cp*cr)*sz,           tz],
                    [0,          0,                   0,                    1 ] 
                   ])

    
    m[:3,:3] = TMP_A @ m[:3,:3] @ TMP_At
    return m

def transform_to_parameters(m : np.array) -> TransformParams:  
    """
    Extract transform to its translation, rotation[degs] and scaling
    Args:
        m   :  transform matrix - 2d numpy homogenoius transform
    Returns:
        list of parameters according to the follow order
        [tx, ty, tz, rx, ry, rz, sx, sy, sz]
        tx - x translation 
        ty - y translation
        tz - z translation
        rx - roll [deg]
        ty - pitch [deg]
        tz - yaw [deg]
        sx - x scale
        sy - y scale
        sz - z scale
    """                
    # first, we decompose the matrix, and extract the scale component. according to https://math.stackexchange.com/questions/237369/given-this-transformation-matrix-how-do-i-decompose-it-into-translation-rotati
    # then, we extract roll/pitch/yaw from the rotation matrix:
    # For carla, rotations order is (1)roll (2)pitch (3)yaw.
    # The axies and rotation directions are as shown in: https://confluence.mobileye.com/display/SG/Coordinates+systems+and+conversions+for+ME+and+Carla
    # The rotation&scale matrix is therefore:
    # (c and s stands for cos and sin, r,p and y stands for roll,pitch and yaw. sx/sy/sz are the scale factors)
    #     |(cy*cp) * sx, (cy*sp*sr-sy*cr) * sy, (-cy*sp*cr-sy*sr) * sz, 0.0|
    # m = |(sy*cp) * sx, (sy*sp*sr+cy*cr) * sy, (-sy*sp*cr+cy*sr) * sz, 0.0|
    #     |sp * sx,      (-cp*sr) * sy,         (cp*cr) * sz,           0.0|        
    #     |0.0,          0.0,                   0.0,                    1.0|        
    #
    # Extracting pitch
    #   pitch = arcsin(m[2,0] / sx), 
    # which gives results in the range [-pi/2, pi/2]. This is ok even if we would initially set pitch 
    # to a value that is out of this range because the other angles roll and yaw will take values to make the overall rotation as needed.
    # (There are different angles values that gives the same overall rotation) 
    #
    # Extracting roll and yaw:
    # we have to consider two cases: when cp=0 and when cp>0  
    # When cp>0 (or sp<1)
    #   roll = atan2(-m[2,1] / sy, m[2,2] / sz)   
    #   yaw = atan2(m[1,0] / sx, m[0,0] / sx)             
    # When cp=0 (or sp =1): This is a special case where yaw and roll change the overall rotation around the same axis Z. 
    # Thus, we can use one of them to have the desired overall rotation while the other is set to 0. 
    # if we set the yaw to 0 (in addition to cp=0, which means that sp=1) the rotation matrix became:
    #     |0.0, sr * sy, -cr * sz,  0.0|
    # m = |0.0, cr * sy,  sr * sz,  0.0|
    #     |1.0, 0.0,      0.0,      0.0|        
    #     |0.0, 0.0,      0.0,      1.0|         
    #   roll = atan2(m[1,2] / sz, m[1,1] / sy)   
    #   yaw = 0             
    
    m[:3,:3] = TMP_At @ m[:3,:3] @ TMP_A

    # extract scale, according to https://math.stackexchange.com/questions/237369/given-this-transformation-matrix-how-do-i-decompose-it-into-translation-rotati
    sx_vec = np.array([m[0,0], m[1,0], m[2,0]])
    sy_vec = np.array([m[0,1], m[1,1], m[2,1]])
    sz_vec = np.array([m[0,2], m[1,2], m[2,2]])
    
    # calc vector magnitudes
    sx = np.sqrt(sx_vec.dot(sx_vec))
    sy = np.sqrt(sy_vec.dot(sy_vec))
    sz = np.sqrt(sz_vec.dot(sz_vec))

    # extract yaw/pitch/roll
    sp = m[2,0] / sx
    
    r = 0.0
    p = math.asin(sp)
    y = 0.0
    
    if sp > 0.9999:
       r = math.atan2(m[1,2] / sz, m[1,1] / sy)   
       y = 0
    else:
       r = math.atan2(-m[2,1] / sy, m[2,2] / sz)   
       y = math.atan2(m[1,0] / sx, m[0,0] / sx)              
    
    #return [m[0,3], m[1,3], m[2,3], r*180.0/math.pi, p*180.0/math.pi, y*180.0/math.pi, sx, sy, sz]
    #def __init__(self, x=0.0,y=0.0,z=0.0,roll=0.0,pitch=0.0,yaw=0.0, sx=1.0, sy=1.0, sz=1.0):

    
    return TransformParams(x=m[0,3],
                                          y=m[1,3],
                                          z=m[2,3],
                                          roll=-r*180.0/math.pi, 
                                          pitch=-p*180.0/math.pi,
                                          yaw=y*180.0/math.pi, 
                                          sx=sx, 
                                          sy=sy, 
                                          sz=sz)


def object_relative_to_object_transform(object1_transform : np.array, object0_transform : np.array ) -> np.array:
    """ create transform of object1 relative to object0

        The transforms object1_transform and object0_transform must be in the same space!
        For example, if object0_transform is in world space, then object1_transform must be in world space as well.
        For example, if object0_transform is relative to some actorX, then object1_transform must be relative to actorX as well.

        Args:
            object1_transform      : Transform of the object1 in the same space as actor 0. 
            object0_transform      : transform of the object0 in the same space as actor 1. 
        Returns:
            transform of object1 relative to object0
    """
        
    # R -  transform of object1 relative to object0
    # T1 - object1 trasnfrom
    # T0 - object0 transform

    # T1 = T0*R
    # R = inv(T0)*T1

    invT0 = np.linalg.inv(object0_transform)
    R = invT0.dot(object1_transform )    
    return R

def object_relative_to_object_transform_parameters(object1_transform_parameters : TransformParams, object0_transform_parameters : TransformParams ) -> np.array:
    """ create transform of object1 relative to object0

        The transforms object1_transform_parameters and object0_transform_parameters must be in the same space!
        For example, if object0_transform_parameters is in world space, then object1_transform_parameters must be in world space as well.
        For example, if object0_transform_parameters is relative to some actorX, then object1_transform_parameters must be relative to actorX as well.

        Args:
            object1_transform_parameters      : sim_types.TransformParams of object1 in the same space as actor 0. 
            object0_transform_parameters      : sim_types.TransformParams of object0 in the same space as actor 1. 
        Returns:
            transform parameters of object1 relative to object0
    """

    object1_transform = parameters_to_transform(object1_transform_parameters)
    object0_transform = parameters_to_transform(object0_transform_parameters)
    obj1_2_obj0 = object_relative_to_object_transform(object1_transform = object1_transform, object0_transform = object0_transform)
    return transform_to_parameters(obj1_2_obj0)

def follower_object_transform(target_object_world_transform : np.array, distance : float, yaw :float, pitch : float, roll : float) -> np.array:
    """ return world transform of follower object. 
    Args:
        target_object_world_transform : 2d 4X4 np matrix homogeniouse transform
        distance                       : distance of follower from target object [meters]
        yaw, pitch, roll               : angle of follower relative to target object [degrees]
    Returns:
        follower world transform 
    """
    # WT0 - target world transform
    # WT1 - follower world transform
    # LT1 - follower Local transform relative to target
    # 
    # WT1 = LT1*WT0
    tmp_rotation_transform = parameters_to_transform(sim_types.TransformParams(0,0,0,roll = roll, pitch = pitch, yaw=yaw)) 
    local_poistion = tmp_rotation_transform.dot(np.array([distance,0,0,1]))
    local_transform = parameters_to_transform(sim_types.TransformParams(local_poistion[0],local_poistion[1],local_poistion[2],roll = roll, pitch = -pitch, yaw=yaw+180))     
    world_tranform  = target_object_world_transform.dot(local_transform)
    return world_tranform


def world_to_local(world_points_3d : np.array, object_world_transform : np.array) ->  np.array:  
    """ Convert world 3d points to an object local 3d points
        Args:
            world_points_3d        : numpy matrix of columns homogenuous world 3D coords
            object_world_transform : numpy matrix of object transform
        Return:
            view points by numpy 2D array of a homogenious 4d vectors in carla space, each column is a point vector [meter]
    """
    # T     = object_world_transform
    # invT  = inv(object_world_transform)
    # world = T*local
    # local = invT*world
    invT = np.linalg.inv(object_world_transform)
    local_points_3d = invT.dot(world_points_3d)
    return local_points_3d

def world_to_local_direction(world_directions_3d, object_world_transform):  
    """
    Convert 3d direction vectors in world space to 3d direction vectors in view space
    Args:
        world_directions_3d    : 2d numpy matrix, contains columns of 3D directions vectors in world space
        object_world_transform : numpy matrix of camera transform
    Return:
        2d numpy matrix, contains columns of 3D directions vectors in view space
    """
    # R     = Rotation(camera_world_transform)
    # invR  = inv(R) = transpose(R), becasue rotation is orthonormal
    # world = R*view
    # view = invR*world
    invR = object_world_transform[:3,:3].transpose()
    view_directions_3d = invR.dot(world_directions_3d)
    return view_directions_3d

def local_to_world(local_points_3d : np.array, object_world_transform : np.array) -> np.array:  
    """
    Convert 3d points in local space to 3d points in world space 
    Args:
        local_points_3d        : numpy matrix of columns homogenuous local 3D coords
        object_world_transform : numpy matrix of camera transform
    Return:
        world points by numpy 2D array of a homogenious 4d vectors in carla space, each column is a point vector [meter]
    """
    # T     = camera_world_transform
    # world = T*local
    world_points_3d = object_world_transform.dot(local_points_3d)
    return world_points_3d

def local_to_world_direction(local_directions_3d, object_world_transform):  
    """
    Convert 3d direction vectors in view space to 3d direction vectors in world space
    Args:
        world_directions_3d    : 2d numpy matrix, contains columns of 3D directions vectors in view space
        object_world_transform : numpy matrix of camera transform
    Return:
        2d numpy matrix, contains columns of 3D directions vectors in world space
    """
    # R     = Rotation(camera_world_transform)
    # world = R*view
    R = object_world_transform[:3,:3]
    world_directions_3d = R.dot(local_directions_3d)
    return world_directions_3d

def normalize_vectors(vectors : np.array) -> np.array:  
    """
    normalize vectors so each has length of 1
    Args:
        vectors    : 2d numpy matrix, contains columns vectors
    Return:
        2d numpy matrix, contains columns unit vectors
    """
    # acis 0, axis of columns
    vector_lengths = np.linalg.norm(vectors, axis=0)
    true_table = vector_lengths == 0.0
    vector_lengths[true_table] = 1.0
    return vectors/vector_lengths


def world_to_image_pinhole(world_points_3d : np.array, camera_world_transform : np.array, camera_projection_matrix : np.array) -> np.array:  
    """
    returns projected 2D coords for 3D coords over image 
    Args:

        world_points_3d          : numpy matrix of columns homogenuous world 3D coords
        camera_world_transform   : numpy matrix of camera transform
        camera_projection_matrix : numpy matrix of pinhole projection of the camera
    Return:
        2d numpy matrix, contains columns of image 2D points 
    """
    # get local coords 
    local_points_3d = world_to_local(world_points_3d = world_points_3d, object_world_transform = camera_world_transform)
    # get the projected 2D
    V = np.dot(camera_projection_matrix, local_points_3d)    
    return np.array([V[0,:]/V[2, :], V[1,:]/V[2, :]]) 

def image_to_world_ray_pinhole(images_points_2d : np.array, camera_world_transform : np.array, camera_projection_matrix : np.array) -> np.array:  
    """
    returns 3d rays for 2D image coords 
    Args:
        images_points_2d         : numpy matrix of columns of image coordinates
                                       [ 
                                         [x0 x1 x2 ... xN],
                                         [y0 y1 y2 ... yN],
                                       ] 
        camera_world_transform   : numpy 4x4 matrix of camera transform
        camera_projection_matrix : numpy 4x4 matrix of pinhole projection of the camera
    Return:
        3d numpy matrix, contains columns of 3d directions
                                       [ 
                                         [x0 x1 x2 ... xN],
                                         [y0 y1 y2 ... yN],
                                         [z0 z1 z2 ... zN],
                                       ] 
    """
    N = images_points_2d.shape[1]
    images_points_homogeniose = np.ones((4,N), dtype = np.float64)
    images_points_homogeniose[:2,:] = images_points_2d
    local_points_3d = np.dot(np.linalg.inv(camera_projection_matrix), images_points_homogeniose)        
    rays = local_points_3d[:3,:]
    return local_to_world_direction(local_directions_3d = rays, object_world_transform = camera_world_transform)

    
def find_front_back_points(world_points_3d : np.ndarray , object_world_transform : np.ndarray, boundary_distance : float) -> Tuple[np.ndarray,np.ndarray] :    
    """ return indexes of world 3d points that are in front of an object, and indexes of those that are behind the object.
        The distance boundary_distance [meters] indicates what is considered front and what is back. 
        Every point that its distance is greater than boundary_distance is considered as front, o.w, it is considered as back 

        Args:
            world_points_3d : numpy 2D array of a homogenious 4d vectors in carla space, each column is a point vector [meter]
                                       [ 
                                         [x0 x1 x2 ... xN],
                                         [y0 y1 y2 ... yN],
                                         [z0 z1 z2 ... zN],
                                         [ 1  1  1 ...  1],
                                       ] 
            cam_world_transform   : numpy 2D array of a homogenious 4d vectors representing camera transform in carla world
            boundary_distance     : Every point that its distance is greater than boundary_distance is considered as front, o.w, it is considered as back.

        Returns:
            [fronts, backs]
            fronts - list of indexes of 3d points that are considered as being in front of the camera
            backs - list of indexes of 3d points that are considered as being at the back of the camera
    """
    # get view coords 
    local_points_3d = world_to_local(world_points_3d, object_world_transform)
    return np.where(local_points_3d[0,:] > boundary_distance)[0] , np.where(local_points_3d[0,:] <= boundary_distance)[0]


def perpendicular(front : np.array,  strive : np.array ) -> np.array:
    """
    Finds a unit vector that is perpendicular to front vector, and aims as much as possible to strive direction.        
    Args:
        front        : a vector point to front
        strive       : a vector aim for direction where the perpendicular should point as much as possible.
    Returns:
        perpendicular vector 
        returns None
    """
    # Note: direction of the cross product vector is defined by the right-hand rule.
    #       because our system is left hand, we correct it by minus 
    v1 = np.cross(front, strive)
    pv = np.cross(front, v1)
    a = np.linalg.norm(pv)
    if 0==a:
        return np.array([np.nan,np.nan,np.nan])
    
    return -pv/a


def rotation_matrix_from_direction(front : np.array,  strive_up : np.array ) -> np.array:
    """
    create rotation matrix from front vector, and up that aims as much as possible to strive_up direction.        
    Args:
        front        : a vector point to front 
        strive       : a vector aim for direction where the up vector should point as much as possible. 
    Returns:
        perpendicular vector 
        returns nan if strive direction is like front direction
    """

    up = perpendicular(front,  strive_up)

    # Note: direction of the cross product vector is defined by the right-hand rule.
    #       because our system is left hand, we correct it by minus 
    right = -np.cross(front, up)

    vx = front/np.linalg.norm(front)
    vy = right/np.linalg.norm(right)
    vz = up/np.linalg.norm(up)

    r = np.zeros((4,4),dtype=np.float64)
    r[3,3]=1
    r[0:3,0] =vx 
    r[0:3,1] =vy 
    r[0:3,2] =vz 
    return r


def look_at_transform(position_point : np.array, look_at_point : np.array, strive_up : np.array):
    """
    create rotation matrix from front vector, and up that aims as much as possible to strive_up direction.        
    Args:
        position_point : a point to look from. Can be 3 elements vector or 4 elements homogeinouse vector 
        look_at_point  : a point to look at. Can be 3 elements vector or 4 elements homogeinouse vector 
        strive         : a direction vector of 3 elements aim for direction where the up vector should point as much as possible. 
    Returns:
        transform contains translation to position_point, and rotation to look at look_at_point
        
    """
    front =  look_at_point[:3] - position_point[:3]
    # rotation
    m =  rotation_matrix_from_direction(front = front,  strive_up = strive_up )
    # translation
    m[:3,3] = position_point[:3]
    
    return m

def look_at_transform_by_direction(look_direction : np.array, look_at_point : np.array, distance : float, strive_up : np.array):
    """
    create transform  matrix from front vector, and up that aims as much as possible to strive_up direction.        
    Args:
        look_direciton : a direction vector of 3 elements which is the looking direction. 
        look_at_point  : a point to look at. Can be 3 elements vector or 4 elements homogeinouse vector 
        distance       : the distance the camera should be from the look_at_point
        strive         : a direction vector of 3 elements aim for direction where the up vector should point as much as possible. 
    Returns:
        transform contains translation and rotation so it look at look_a_point
    """
    # Calculate the norm of the vector
    norm = np.linalg.norm(look_direction)

    # Normalize the vector
    if norm != 0:
        look_direction_unit = look_direction / norm
    else:
        look_direction_unit = look_direction  # If norm is zero, the vector is already ze
            
    position_point = look_at_point[:3] - look_direction_unit*distance
    return look_at_transform(position_point=position_point, look_at_point=look_at_point, strive_up = strive_up)


def rays_plane_intersection(rays_position_3d : np.array,  rays_direction_3d : np.array, plane_transform : np.array) -> np.array:
    """
    find intersection points of rays and plane
    Args:
        rays_position_3d: 2d numpy matrix, contains columns of each ray starting world 3d point. 
                                       [ 
                                         [x0 x1 x2 ... xN],
                                         [y0 y1 y2 ... yN],
                                         [z0 z1 z2 ... zN],
                                         [1  1  1 ...  1],
                                       ]  
        rays_direction_3d: 2d numpy matrix, contains columns of each ray direction. 
                                       [ 
                                         [x0 x1 x2 ... xN],
                                         [y0 y1 y2 ... yN],
                                         [z0 z1 z2 ... zN],
                                       ]  
        plane_transform : transform of plane 
    Returns:
        2d numpy matrix, contains interestion points
    """
    #  p0 - plane position
    #  n  - plane normal
    #  r0 - ray position 
    #  r  - ray direction 
    #
    #  X - intersection point]
    #  X = r0 + r*t 
    #
    # plane normal is perpendicular to X-p0  
    # (X-p0)*n = 0
    # (r0 + r*t - p0)*n = 0
    # t(r*n) = (p0-r0)*n
    # t = (p0-r0)*n / t(r*n)
    
    # plane position 
    p0 = plane_transform[:3,3]

    # plane normal is the world direciton of its z local. 
    # this i is pR.dot(np.array([[0],
    #                            [0],
    #                            [1]]))
    n = plane_transform[:3,2]

    A = (np.array([p0]).T - rays_position_3d[:3,:])
    noms = n.dot(A)
    denoms = n.dot(rays_direction_3d)
    valid_indices =  np.where(abs(denoms) > 1e-6)[0]    
    t = noms[valid_indices] / denoms[valid_indices]


    out = np.empty((4,rays_position_3d.shape[1]), dtype = np.float64)
    out[:] = np.nan
    out[:,valid_indices] = rays_position_3d[:,valid_indices]
    out[:3,valid_indices] = out[:3,valid_indices] + np.array([t,t,t])*rays_direction_3d[:,valid_indices]

    return out

    
def rotation_vector_to_rotation_matrix(rotation_vector : np.array) -> np.array:
    """
    convert rodrigus rotation vector (v=θk) to rotation matrix
    Args:
        rotation_vector : v=θk - 3d vector where its direction is the rotation axis, and its length is the rotation angle in radians.
                                 positive rotation angle is CCW when looking down the axis toward origin
    Returns:
        4x4 homogeniouse rotation matrix
    """
    # R=cos(θ)∙I+sin(θ)∙K+(1-cos(θ))∙(k∙k^T )
    #
    #            |   0  kz -ky |
    # where K =  | -kz   0  kx |   
    #            |  ky -kx   0 |
    #            
    angle = np.linalg.norm(rotation_vector)
    if abs(angle) < 1e-6:
        return np.eye(4, dtype=np.float64) 
    k = rotation_vector/angle
    kx = k[0]
    ky = k[1]
    kz = k[2]       
    K = np.array([[    0,  kz, -ky],
                  [  -kz,   0,  kx],
                  [   ky, -kx,   0]])
    I = np.identity(3, dtype=np.float64)
    R3 = math.cos(angle)*I + math.sin(angle)*K + (1-math.cos(angle))*(k.reshape(3,1).dot(k.reshape(1,3)))
    R = np.identity(4, dtype=np.float64)
    R[:3,:3] = R3
    return R

def remove_scale_from_transform(transform : np.array) -> np.array:
    """
    remove scale from transform.
    Function assume positive scale. 
    If scale is negative, the function results may be incorrect.
    Is scale near zero, the function raises an error.
    Args:
        transform : 4x4 homogeniouse transform matrix
    Returns:
        transform : 4x4 homogeniouse transform matrix with no scale
    """
    T = transform.astype(np.float64, copy=True)

    scales = np.linalg.norm(T[:3, :3], axis=0)
    if np.any(scales <= 1e-7):
            raise ValueError("Degenerate scale: cannot recover rotation (axis collapsed).")
    T[:3,:3] /= scales

    # Optional: enforce orthogonality via QR
    #Q, _ = np.linalg.qr(T[:3,:3])
    # enforce proper rotation (det = +1)
    #if np.linalg.det(Q) < 0:
    #    Q[:, 2] *= -1
    #T[:3,:3] = Q

    return T


def rotation_matrix_to_rotation_vector(rotation_matrix : np.array) -> np.array:
    """
    convert rotation matrix to  rodrigus rotation vector (v=θk) 
    Args:
        rotation_matirx : 4x4 homogeniouse (translation is ignored) or 3x3 rotation matrix
    Returns:
        rotation vector - 3d vector where its direction is the rotation axis, and its length is the rotation angle in radians.
                          positive rotation angle is CCW when looking down the axis toward origin
    """
    # if rotation matrix contains scale 
    #if np.linalg.norm(rotation_matrix[:3,:3] - np.eye(3)) > 1e-6:
    #    raise ValueError("Rotation matrix has scale")


    angle = math.acos(clamp_val((np.trace(rotation_matrix[:3,:3])-1)/2.0, -1.0, 1.0))
    # if angle is zero, then the rotation vector is arbitrary, so we chose X axis
    if abs(angle) < 1e-6:
        return np.array([0.0,0.0,0.0])
    
    # if angle is pi or -pi, then the rotation vector should be found in different way
    # no important for sign, because for 180 degrees rotation, the axis can be in two opposite directions
    if abs(abs(angle)-math.pi) < 1e-6:
        kx = math.sqrt((rotation_matrix[0,0]+1)/2.0)
        ky = math.sqrt((rotation_matrix[1,1]+1)/2.0)
        kz = math.sqrt((rotation_matrix[2,2]+1)/2.0)
        k = np.array([kx, ky, kz])
        # find relative signs of each component:
        # find index of largest component, we assume this component is positive
        max_index = np.argmax(k)        
        # Method 1:
        # set sign relative to the largest component. sign of 0 will be set as 1        
        k_relative_signs = np.where(rotation_matrix[:3,max_index]/k[max_index] < 0, -1, 1)
        k = k * k_relative_signs
        # Method 2:
        # off disgonal R values are 2*k[i]*k[j] for angle of 180 degrees
        #k_max= k[max_index]
        #k = rotation_matrix[:3,max_index]/(2*k[max_index])
        #k[max_index] = k_max
    else:
        # the normal way 0<angle<pi  or -pi<angle<0
        kx = rotation_matrix[1,2] - rotation_matrix[2,1]
        ky = rotation_matrix[2,0] - rotation_matrix[0,2]
        kz = rotation_matrix[0,1] - rotation_matrix[1,0]
        k = np.array([kx, ky, kz])
    
    k = k/np.linalg.norm(k)
    return k * angle

def rotation_quaternion_to_rotation_vector(quaternion : np.array, short_path = False) -> np.array:
    """
    converts rotation quaternion to rotation vector
    Args:
        quaternion : (w,x,y,z) where 
                      w = cos(θ/2),  
                      x,y,z=v*sin(θ/2), 
                      v - unit vector of the rotation axis, 
                      θ - rotation angle in radians, positive rotation angle is CCW when looking down the axis toward origin
        short_path : if True, the quaternion will be taken with the smaller angle path
                     if False, the quaternion will be taken with the larger angle path
                     default is False
    Returns:
        rotation vector : 3d vector where its direction is the rotation axis, and its length is the rotation angle in radians.
    """
    # validation - check that norm of quaterion is approximatly 1
    if abs(np.linalg.norm(quaternion) - 1) > 1e-6:
        raise ValueError("Quaternion is not a valid rotation quaternion")
    # take the quternion form with smaller angle path
    if short_path:
        if quaternion[0] < 0:
            quaternion = -quaternion
    # convert quternion to rotation vectorץ    
    # clamp the quaternion[0] to be between -1 and 1 in order to avoid numerical errors
    θ = 2*math.acos(np.clip(quaternion[0], -1.0, 1.0))
    if abs(θ) < 1e-6:
        return np.array([0,0,0], dtype=np.float64) 
    else:   
        v = quaternion[1:]/math.sin(θ/2)
    return v * θ

def rotation_vector_to_rotation_quaternion(rotation_vector : np.array) -> np.array:
    """
    converts rotation vector to rotation quaternion
    Args:
        rotation_vector : 3d vector where its direction is the rotation axis, and its length is the rotation angle in radians.
    Returns:
        quaternion : (w,x,y,z) where 
                      w = cos(θ/2),  
                      x,y,z=v*sin(θ/2), 
                      v - unit vector of the rotation axis, 
                      θ - rotation angle in radians, positive rotation angle is CCW when looking down the axis toward origin
    """
    θ = np.linalg.norm(rotation_vector)
    if abs(θ) < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])
    else:
        v = rotation_vector/θ
        return np.array([math.cos(θ/2), v[0]*math.sin(θ/2), v[1]*math.sin(θ/2), v[2]*math.sin(θ/2)])

def rotation_quaternion_to_rotation_matrix(quaternion : np.array) -> np.array:
    """
    converts quaternion to rotation matrix
    Args:
        quaternion : (w,x,y,z) where 
                      w = cos(θ/2),  
                      x,y,z=v*sin(θ/2), 
                      v - unit vector of the rotation axis, 
                      θ - rotation angle in radians, positive rotation angle is CCW when looking down the axis toward origin
    Returns:
        4x4 homogeniouse rotation matrix
    """
    # convert quternion to rotation vector
    rotation_vector = rotation_quaternion_to_rotation_vector(quaternion)
    # convert rotation vector to rotation matrix
    return rotation_vector_to_rotation_matrix(rotation_vector)


def rotation_matrix_to_rotation_quaternion(rotation_matrix : np.array) -> np.array:
    """
    converts rotation matrix to quaternion
    Args:
        rotation_matrix : 4x4 homogeniouse rotation matrix
    Returns:
        quaternion : (w,x,y,z) where 
                      w = cos(θ/2),  
                      x,y,z=v*sin(θ/2), 
                      v - unit vector of the rotation axis, 
                      θ - rotation angle in radians, positive rotation angle is CCW when looking down the axis toward origin
    """
    # convert rotation matrix to rotation vector
    rotation_vector = rotation_matrix_to_rotation_vector(rotation_matrix)
    # convert rotation vector to quaternion
    return rotation_vector_to_rotation_quaternion(rotation_vector)


def get_rotation_angles_by_order(m : np.ndarray, order : str) -> [float, float, float]:
    """
    get rotation angles (degrees) according to desired order
    Args:
        m : 4x4 homogeniouse transform matrix
        order : rotation order, 
               'rpy' - roll, pitch, yaw
               'ryp' - roll, yaw, pitch
               'pry' - pitch, roll, yaw
               'pyr' - pitch, yaw, roll
               'yrp' - yaw, roll, pitch
               'ypr' - yaw, pitch, roll
    Returns:
        rotation angles (degrees) according to desired order
    """
    # in scipy: 
    # We doesn'y use roll, pitch yaw terms because roll pitch and yaw are assigned differnetly on each application. 
    #    for example, in ME roll is around Z, while in SIm it is around X.
    # Instead, scipy uses rotations around axis: x,y, and z (then MEANING of it, roll, pitch or yaw defined by the application) 
    # Scipy use the left hand rule, thus: 
    #    positive angle is counterclockwise around axis when looking toward the axis heading.
    #
    # Thus, for sim, where x is roll, y is pitch and z is yaw:      
    # scipy positive angle around x is counterclockwise , while sim positive roll is clockwise
    # scipy positive angle around y is counterclockwise , while sim positive yaw is clockwise (up)
    # scipy positive angle around z is counterclockwise , and sim positive yaw is counterclockwise too (heading right)
    # thus
    # roll_sim = -roll_scipy
    # pitch_sim = -pitch_scipy
    # yaw_sim = yaw_scipy

    # replace 
    order_xyz = order.replace('y', 'z').replace('p', 'y').replace('r', 'x')

    # convert rotation matrix to rotation vector
    rot = Rotation.from_matrix(m[:3,:3])
    res = rot.as_euler(order_xyz, degrees=True)
    r_index = order_xyz.index('x')
    p_index = order_xyz.index('y')
    y_index = order_xyz.index('z')
    r = res[r_index]
    p = res[p_index]
    y = res[y_index]
    return [-r,-p,y]


def convert_rotation_angles_for_order(rpy : [float, float, float], order : str) -> [float, float, float]:
        """
        convert values of rotation angles for different order
        Args:
            rpy : rotation angles (degrees) according  to order of roll->pitch->yaw
            desired order :
                   'rpy' - roll, pitch, yaw
                   'ryp' - roll, yaw, pitch
                   'pry' - pitch, roll, yaw
                   'pyr' - pitch, yaw, roll
                   'yrp' - yaw, roll, pitch
                   'ypr' - yaw, pitch, roll
        """
        m_params = TransformParams(roll = rpy[0], pitch = rpy[1], yaw = rpy[2])
        m = parameters_to_transform(m_params)
        rpy_new = get_rotation_angles_by_order(m, order)
        return rpy_new


