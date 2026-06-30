from __future__ import annotations
from typing import Callable, Dict, List, Optional, Union
from pathlib import Path
import itertools
import threading
import collections
import math
import numpy as np

"""
Design
==================================================
general
---------
1. IMPORTANT:grx tends to be realtime 3d engine, so it should be fast and efficient.
2. grx will be based on Panda3D engine. but can be changed if there is need to better performance.
3. Coordinates system in grx is according to "\\grx\\source\\grx\\docs\\coordinates system.md"
4. Units are meters and degrees.
5. Colors are in RGBA format (0.0-1.0) for example: [1.0, 0.0, 0.0, 1.0] is red.
6. transparency is supported. but can be disabled per object., or per scene.
7. use numpy for efficient math library for calculations. can be changed if there is need to better performance.
8. transform is 4x4 matrix in column major order.
   this means, transforming a homogeniouse vector point is done by transform_matrix @ vector_point
9. every SceneObject is dictionaried by unique name
   if name is not given , it automatically generated uniqued name
   if name is given, if it is not unique, it will be appended with a index to make it unique.
10. Functions that tends to be called from player.update() function must be synced within the rendering thread
    in order to allow fast and smooth rendering and responding to user input,
    do not wait , but push commands to a queue and let the rendering thread handle them.


Architecture - threading & the command queue
---------------------------------------------
To keep navigation/rendering fluid (even with two viewers) and to respond fast to
user keyboard/mouse, grx uses a single rendering thread (the thread that runs
``player.run()``) plus a thread-safe *command queue*.

* The **data model** (Scene graph + numpy transforms) is the single source of
  truth. It is updated immediately (under a lock) so reads such as
  ``get_transform()`` are non-blocking, immediate and always consistent. This is
  what the unit tests exercise, and it does not require a window/engine.

* The **visual layer** (Panda3D nodes, camera, lights, background, snapshots) must
  only be touched by the rendering thread. Every mutating call that affects the
  visual layer therefore does NOT touch Panda3D directly. Instead it pushes a
  command (a callable) onto the engine's command queue and returns immediately.
  Once per frame the rendering thread drains the queue and applies the commands,
  then runs ``player._update()``, then renders all viewers.

This single mechanism (the command queue) is used *consistently* for every
operation that needs to be synced with the rendering thread, so callers - whether
they run on the render thread (inside ``player._update()``) or on another thread -
never block the renderer.

run_scene
----------
1. run_scene is the main function that will be used to run the scene.
2. User navigation should be fluid and natural. without lags.
"""

# =====================================================================
# Coordinates system (see docs/coordinates system.md)
#   left handed:  x = right, y = up, z = front
# Panda3D default is right handed Z-up: x = right, y = forward, z = up.
# Converting grx <-> panda is a swap of the y and z axes (which also flips
# handedness, mapping the left handed grx frame to the right handed panda frame).
# =====================================================================

Color = List[float]

WHITE = [1.0, 1.0, 1.0, 1.0]
LIGHT_GRAY = [0.75, 0.75, 0.75, 1.0]
RED = [1.0, 0.0, 0.0, 1.0]
GREEN = [0.0, 1.0, 0.0, 1.0]
BLUE = [0.0, 0.0, 1.0, 1.0]

# change of basis grx <-> panda (swap y and z). It is its own inverse.
_GRX_PANDA_BASIS = np.array([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)


def hello():
    print("Hello world!")


# =====================================================================
# Transform helpers (column major, point transform is  M @ vector)
# =====================================================================
def identity_transform() -> np.ndarray:
    """Return a 4x4 identity transform."""
    return np.eye(4, dtype=np.float64)


def as_transform(transform: np.ndarray) -> np.ndarray:
    """Validate and return a contiguous float64 copy of a 4x4 transform."""
    t = np.asarray(transform, dtype=np.float64)
    if t.shape != (4, 4):
        raise ValueError(f"transform must be a 4x4 matrix, got shape {t.shape}")
    return t.copy()


def translation_transform(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    """Return a 4x4 transform that translates by (x, y, z) [meters]."""
    t = np.eye(4, dtype=np.float64)
    t[0, 3] = x
    t[1, 3] = y
    t[2, 3] = z
    return t


def axis_angle_transform(axis: List[float], angle_deg: float) -> np.ndarray:
    """Return a 4x4 rotation transform of ``angle_deg`` degrees about ``axis``.

    Uses the Rodrigues formula. ``axis`` need not be normalized.
    """
    v = np.asarray(axis, dtype=np.float64)[:3]
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return np.eye(4, dtype=np.float64)
    k = v / norm
    theta = math.radians(angle_deg)
    kx, ky, kz = k
    K = np.array([[0.0, -kz, ky],
                  [kz, 0.0, -kx],
                  [-ky, kx, 0.0]], dtype=np.float64)
    R3 = np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)
    R = np.eye(4, dtype=np.float64)
    R[:3, :3] = R3
    return R


def compose(*transforms: np.ndarray) -> np.ndarray:
    """Compose transforms left to right: compose(A, B, C) == A @ B @ C."""
    result = np.eye(4, dtype=np.float64)
    for t in transforms:
        result = result @ np.asarray(t, dtype=np.float64)
    return result


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    """Return the inverse of a 4x4 transform."""
    return np.linalg.inv(np.asarray(transform, dtype=np.float64))


def relative_transform(object_transform: np.ndarray, reference_transform: np.ndarray) -> np.ndarray:
    """Return ``object_transform`` expressed relative to ``reference_transform``.

    Both transforms must be given in the same space.
    result = inv(reference_transform) @ object_transform
    """
    return np.linalg.inv(np.asarray(reference_transform, dtype=np.float64)) @ np.asarray(object_transform, dtype=np.float64)


def _grx_to_panda_mat4(transform: np.ndarray):
    """Convert a grx column-major 4x4 transform to a Panda3D ``LMatrix4f``.

    The basis is swapped (grx -> panda) and the matrix is transposed because
    Panda3D uses the row-vector convention (point_row * M).
    """
    from panda3d.core import LMatrix4f
    panda = _GRX_PANDA_BASIS @ np.asarray(transform, dtype=np.float64) @ _GRX_PANDA_BASIS
    row_major = np.ascontiguousarray(panda.T)
    return LMatrix4f(*row_major.flatten().tolist())


# =====================================================================
# Scene objects
# =====================================================================
class SceneObject:
    """Base class for a scene object.

    A scene object stores its *local* transform (relative to its parent, or to the
    scene root when it has no parent). The world transform is computed by walking
    up the parent chain. Geometry is created lazily by the rendering thread.

    Parameters:
        name           : unique name. If None an automatic unique name is generated.
                         If not unique, an index is appended to make it unique.
        transform      : 4x4 local transform (relative to parent / scene).
        parent_object  : optional parent SceneObject.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 transform: Optional[np.ndarray] = None,
                 parent_object: Optional["SceneObject"] = None):
        self._name: Optional[str] = name
        self.transform_local: np.ndarray = identity_transform() if transform is None else as_transform(transform)
        self._parent: Optional[SceneObject] = parent_object
        self._children: List[SceneObject] = []
        self._scene: Optional[Scene] = None
        self._visible: bool = True
        # Panda3D NodePath, created by the rendering thread. None until then.
        self._nodepath = None

    # ---- identity -------------------------------------------------------
    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def parent(self) -> Optional["SceneObject"]:
        return self._parent

    @property
    def children(self) -> List["SceneObject"]:
        return list(self._children)

    # ---- transforms (immediate, non blocking reads) ---------------------
    def world_transform(self) -> np.ndarray:
        """Return this object's transform relative to the scene root (world)."""
        if self._parent is None:
            return self.transform_local.copy()
        return self._parent.world_transform() @ self.transform_local

    def _resolve_reference(self, reference_object: Union["SceneObject", str, None]) -> Optional["SceneObject"]:
        if reference_object is None:
            return None
        if isinstance(reference_object, SceneObject):
            return reference_object
        if self._scene is None:
            raise ValueError("object is not part of a scene, cannot resolve reference by name")
        return self._scene.get_object(reference_object)

    def get_transform(self, reference_object: Union["SceneObject", str, None] = None) -> np.ndarray:
        """Get the transform of this object relative to the reference object.

        reference_object can be a SceneObject, a string name, or None.
        If None, the transform is relative to the scene (world).
        This read is immediate and non blocking.
        """
        world = self.world_transform()
        reference = self._resolve_reference(reference_object)
        if reference is None:
            return world
        return relative_transform(world, reference.world_transform())

    def set_transform(self,
                       transform: np.ndarray,
                       reference_object: Union["SceneObject", str, None] = None) -> None:
        """Set the transform of this object relative to the reference object.

        reference_object can be a SceneObject, a string name, or None.
        If None, the transform is relative to the scene (world).

        The data model is updated immediately; a command to update the visual
        node is pushed to the rendering thread's queue.
        """
        transform = as_transform(transform)
        reference = self._resolve_reference(reference_object)
        if reference is None:
            target_world = transform
        else:
            target_world = reference.world_transform() @ transform

        if self._parent is None:
            self.transform_local = target_world
        else:
            self.transform_local = relative_transform(target_world, self._parent.world_transform())

        if self._scene is not None:
            self._scene._enqueue(lambda: self._apply_transform_to_node())

    # ---- backward compatible aliases (stub spelling) --------------------
    def get_tranform(self, reference_object: Union["SceneObject", str, None] = None) -> np.ndarray:
        return self.get_transform(reference_object)

    def set_tranform(self,
                     transform: np.ndarray,
                     reference_object: Union["SceneObject", str, None] = None) -> None:
        return self.set_transform(transform, reference_object)

    # ---- visual layer (rendering thread only) ---------------------------
    def _build_nodepath(self):
        """Build and return a Panda3D NodePath for this object.

        Subclasses with geometry override this. The base object is an empty node.
        Runs on the rendering thread only.
        """
        from panda3d.core import PandaNode, NodePath
        return NodePath(PandaNode(self._name or "scene_object"))

    def _apply_transform_to_node(self) -> None:
        """Apply the current local transform to the Panda3D node (render thread)."""
        if self._nodepath is None:
            return
        self._nodepath.setMat(_grx_to_panda_mat4(self.transform_local))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self._name!r})"


class Cube(SceneObject):
    """A cube scene object.

    size  : [width(x), height(y), depth(z)] in meters.
    color : RGBA in 0.0-1.0.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 size: Optional[List[float]] = None,
                 color: Optional[Color] = None):
        super().__init__(name=name)
        self.size: List[float] = list(size) if size is not None else [1.0, 1.0, 1.0]
        self.color: Color = list(color) if color is not None else list(WHITE)

    def _build_nodepath(self):
        return _make_box_nodepath(self._name or "cube", self.size, self.color)


class Arrow(SceneObject):
    """An arrow scene object pointing along +z (front).

    size  : [width(x), height(y), depth/length(z)] in meters.
    color : RGBA in 0.0-1.0.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 size: Optional[List[float]] = None,
                 color: Optional[Color] = None):
        super().__init__(name=name)
        self.size: List[float] = list(size) if size is not None else [0.1, 0.1, 1.0]
        self.color: Color = list(color) if color is not None else list(WHITE)

    def _build_nodepath(self):
        return _make_arrow_nodepath(self._name or "arrow", self.size, self.color)


# kept lowercase too for compatibility with the stub spelling
arrow = Arrow


class Axes(SceneObject):
    """A 3 axes gizmo. Length and colors are customizable.

    Default length is 1 meter.
    Default colors are: x - red, y - green, z - blue.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 length: float = 1.0,
                 x_color: Optional[Color] = None,
                 y_color: Optional[Color] = None,
                 z_color: Optional[Color] = None):
        super().__init__(name=name)
        self.length = float(length)
        self.x_color = list(x_color) if x_color is not None else list(RED)
        self.y_color = list(y_color) if y_color is not None else list(GREEN)
        self.z_color = list(z_color) if z_color is not None else list(BLUE)

    def _build_nodepath(self):
        return _make_axes_nodepath(self._name or "axes", self.length,
                                   self.x_color, self.y_color, self.z_color)


# kept lowercase too for compatibility with the stub spelling
axes = Axes


# =====================================================================
# Cameras
# =====================================================================
class Camera(SceneObject):
    """A camera scene object.

    Looks along +z (front) with +y up, following the grx coordinate system.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 transform: Optional[np.ndarray] = None,
                 parent_object: Optional[SceneObject] = None,
                 width: int = 600,
                 height: int = 400):
        super().__init__(name=name, transform=transform, parent_object=parent_object)
        self.width = int(width)
        self.height = int(height)


class PinHoleCamera(Camera):
    """A pin-hole camera model.

    focal : focal length in pixels.
    ppx   : principal point x (defaults to width / 2).
    ppy   : principal point y (defaults to height / 2).
    """

    def __init__(self,
                 name: Optional[str] = None,
                 transform: Optional[np.ndarray] = None,
                 parent_object: Optional[SceneObject] = None,
                 width: int = 600,
                 height: int = 400,
                 focal: float = 400.0,
                 ppx: Optional[float] = None,
                 ppy: Optional[float] = None):
        super().__init__(name=name, transform=transform, parent_object=parent_object,
                         width=width, height=height)
        self.focal = float(focal)
        self.ppx = float(ppx) if ppx is not None else width / 2.0
        self.ppy = float(ppy) if ppy is not None else height / 2.0

    def projection_matrix(self) -> np.ndarray:
        """Return the 3x3 pin-hole intrinsics matrix K."""
        return np.array([[self.focal, 0.0, self.ppx],
                         [0.0, self.focal, self.ppy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def vertical_fov_deg(self) -> float:
        """Return the vertical field of view in degrees."""
        return math.degrees(2.0 * math.atan2(self.height / 2.0, self.focal))


# =====================================================================
# Lights
# =====================================================================
class SceneLightObject(SceneObject):
    """Base class for a scene light object."""

    def __init__(self,
                 name: Optional[str] = None,
                 color: Optional[Color] = None,
                 transform: Optional[np.ndarray] = None,
                 parent_object: Optional[SceneObject] = None):
        super().__init__(name=name, transform=transform, parent_object=parent_object)
        self.color: Color = list(color) if color is not None else list(WHITE)


class SunLightObject(SceneLightObject):
    """A directional (sun) light. Shines along its local +z (front)."""

    def _build_nodepath(self):
        from panda3d.core import DirectionalLight, NodePath
        light = DirectionalLight(self._name or "sun")
        light.setColor(tuple(self.color))
        return NodePath(light)


# =====================================================================
# Scene
# =====================================================================
class Scene:
    """The scene graph: a registry of uniquely named SceneObjects.

    The scene owns the authoritative data model. Mutations update the model
    immediately (under a lock) and, when an engine is running, push a command
    to the rendering thread to sync the visual node.
    """

    def __init__(self, name: str = "scene"):
        self.name = name
        self._objects: Dict[str, SceneObject] = {}
        self._roots: List[SceneObject] = []
        self._lock = threading.RLock()
        self._auto_counter = itertools.count()
        # set by the player/engine while rendering is active
        self._engine: Optional["_RenderEngine"] = None
        self._render_root = None  # Panda3D NodePath, created by the engine

    # ---- naming ---------------------------------------------------------
    def _unique_name(self, name: Optional[str]) -> str:
        if name is None:
            name = f"object_{next(self._auto_counter)}"
        if name not in self._objects:
            return name
        index = 1
        while f"{name}_{index}" in self._objects:
            index += 1
        return f"{name}_{index}"

    # ---- lookup ---------------------------------------------------------
    def get_object(self, name: str) -> SceneObject:
        with self._lock:
            if name not in self._objects:
                raise KeyError(f"no object named {name!r} in scene {self.name!r}")
            return self._objects[name]

    def objects(self) -> List[SceneObject]:
        with self._lock:
            return list(self._objects.values())

    def _resolve(self, obj: Union[SceneObject, str, None]) -> Optional[SceneObject]:
        if obj is None:
            return None
        if isinstance(obj, SceneObject):
            return obj
        return self.get_object(obj)

    # ---- command queue helper ------------------------------------------
    def _enqueue(self, command: Callable[[], None]) -> None:
        """Push a visual-sync command to the rendering thread if one is active."""
        engine = self._engine
        if engine is not None:
            engine.enqueue(command)

    # ---- mutations (API callable from player.update) -------------------
    def spawn_object(self,
                     object: SceneObject,
                     transform: Optional[np.ndarray] = None,
                     parent_object: Union[SceneObject, str, None] = None) -> SceneObject:
        """Spawn an object into the scene.

        transform is relative to the parent object if given, otherwise relative
        to the scene. Returns the spawned object (with its final unique name).
        """
        if not isinstance(object, SceneObject):
            raise TypeError("spawn_object expects a SceneObject")
        with self._lock:
            name = self._unique_name(object._name)
            object._name = name
            object._scene = self
            if transform is not None:
                object.transform_local = as_transform(transform)
            parent = self._resolve(parent_object)
            object._parent = parent
            if parent is None:
                self._roots.append(object)
            else:
                if object not in parent._children:
                    parent._children.append(object)
            self._objects[name] = object

        self._enqueue(lambda: self._engine.create_visual(self, object))
        return object

    def remove_object(self, object: Union[SceneObject, str]) -> None:
        """Remove an object (and its children) from the scene."""
        with self._lock:
            obj = self._resolve(object)
            if obj is None or obj._name not in self._objects:
                return
            removed = self._collect_subtree(obj)
            for node in removed:
                self._objects.pop(node._name, None)
            if obj._parent is not None:
                if obj in obj._parent._children:
                    obj._parent._children.remove(obj)
            elif obj in self._roots:
                self._roots.remove(obj)
            obj._parent = None

        self._enqueue(lambda: self._engine.remove_visual(removed))

    def _collect_subtree(self, obj: SceneObject) -> List[SceneObject]:
        result: List[SceneObject] = []
        stack = [obj]
        while stack:
            node = stack.pop()
            result.append(node)
            stack.extend(node._children)
        return result


# =====================================================================
# Viewer
# =====================================================================
class Viewer:
    """A window that renders a scene.

    A viewer owns a window (and optionally a camera). Visual-affecting calls are
    synced with the rendering thread through the command queue, so they can be
    called from ``player.update()`` without blocking.
    """

    def __init__(self,
                 name: str,
                 scene: Scene,
                 camera: Union[Camera, str, None] = None):
        self.name = name
        self.scene = scene
        self.camera = camera
        self.background_color: Color = list(WHITE)
        self.grid_size: float = 1.0
        self.grid_color: Color = list(LIGHT_GRAY)
        self.grid_visible: bool = False
        self._enabled: bool = True
        self._engine: Optional["_RenderEngine"] = None
        # Panda3D window / display objects, created by the engine.
        self._window = None
        self._camera_np = None
        self._grid_np = None

    def _enqueue(self, command: Callable[[], None]) -> None:
        engine = self._engine
        if engine is not None:
            engine.enqueue(command)

    # ---- API callable from player.update -------------------------------
    def set_background_color(self, color: Color) -> None:
        """Set the window background color (default white)."""
        self.background_color = list(color)
        self._enqueue(lambda: self._engine.apply_background(self))

    def set_grid(self, size: float, color: Optional[Color] = None) -> None:
        """Set the floor grid (xz plane) cell size and color.

        Default is 1 meter and light gray.
        """
        self.grid_size = float(size)
        if color is not None:
            self.grid_color = list(color)
        self._enqueue(lambda: self._engine.apply_grid(self))

    def set_grid_visible(self, visible: bool = True) -> None:
        """Show or hide the floor grid."""
        self.grid_visible = bool(visible)
        self._enqueue(lambda: self._engine.apply_grid(self))

    def enable(self, enable: bool) -> None:
        """Show (True) or hide (False) the viewer."""
        self._enabled = bool(enable)
        self._enqueue(lambda: self._engine.apply_enabled(self))


# kept lowercase too for compatibility with the stub spelling
viewer = Viewer


# =====================================================================
# User navigators
# =====================================================================
class UserNavigator:
    """Base class for a user navigator.

    A navigator is assigned to a camera object and controls that camera object
    in response to user input. Its ``update`` method is called by the player each
    frame, on the rendering thread.
    """

    def __init__(self, camera_object: Union[Camera, str, None] = None):
        self.camera_object = camera_object

    def update(self, engine: "_RenderEngine", dt: float) -> None:
        """Update the controlled camera. Implemented by subclasses."""
        pass


class ShooterUserNavigator(UserNavigator):
    """
    mouse right - yaw right
    mouse left  - yaw left
    mouse forward - pitch up
    mouse back  - pitch down
    W - moving forward in camera look direction
    S - moving backward to camera look direction
    A - moving left to camera look direction
    D - moving right to camera look direction
    Q - moving up in world up direction
    E - moving down in world up direction
    * rate of movement, and rotation is customizable at initialization.
    """

    def __init__(self,
                 camera_object: Union[Camera, str, None] = None,
                 move_rate: float = 5.0,
                 rotate_rate: float = 90.0,
                 mouse_sensitivity: float = 5.0):
        super().__init__(camera_object)
        self.move_rate = float(move_rate)          # meters / second
        self.rotate_rate = float(rotate_rate)      # degrees / second
        self.mouse_sensitivity = float(mouse_sensitivity)  # degrees / mouse unit
        self.yaw_deg = 0.0
        self.pitch_deg = 0.0

    def update(self, engine: "_RenderEngine", dt: float) -> None:
        camera = engine.resolve_object(self.camera_object)
        if camera is None:
            return

        is_down = engine.is_key_down
        # rotation from mouse delta
        dx, dy = engine.mouse_delta()
        self.yaw_deg += dx * self.mouse_sensitivity
        self.pitch_deg += -dy * self.mouse_sensitivity
        self.pitch_deg = max(-89.0, min(89.0, self.pitch_deg))

        # orientation: yaw about world up (y), then pitch about local right (x)
        rot = compose(axis_angle_transform([0.0, 1.0, 0.0], self.yaw_deg),
                      axis_angle_transform([1.0, 0.0, 0.0], self.pitch_deg))

        world = camera.world_transform()
        position = world[:3, 3].copy()

        forward = rot[:3, 2]
        right = rot[:3, 0]
        world_up = np.array([0.0, 1.0, 0.0])

        step = self.move_rate * dt
        if is_down("w"):
            position += forward * step
        if is_down("s"):
            position -= forward * step
        if is_down("d"):
            position += right * step
        if is_down("a"):
            position -= right * step
        if is_down("q"):
            position += world_up * step
        if is_down("e"):
            position -= world_up * step

        new_world = rot.copy()
        new_world[:3, 3] = position
        camera.set_transform(new_world, reference_object=None)


# =====================================================================
# Player
# =====================================================================
class player:
    """Base class for a player.

    The player updates the scene and renders it through one or more viewers. It
    attaches user navigators to camera objects, can save snapshots, and runs the
    scene. Subclasses implement ``_update`` (called every frame on the rendering
    thread).
    """

    def __init__(self, viewers: List[Viewer]):
        if isinstance(viewers, Viewer):
            viewers = [viewers]
        self.viewers: Dict[str, Viewer] = {v.name: v for v in viewers}
        self._navigators: List[UserNavigator] = []
        self._engine: Optional["_RenderEngine"] = None

    # ---- to be implemented by subclasses -------------------------------
    def _update(self) -> None:
        """Update scene, viewers, objects.

        Functions called here must be synced with the rendering thread (use the
        scene/viewer/player API which pushes to the command queue). Implemented by
        the inheriting class.
        """
        raise NotImplementedError("update must be implemented by inherent class.")

    # ---- run ------------------------------------------------------------
    def run(self) -> None:
        """Run the scene. Blocks on the rendering loop until the window closes."""
        self._engine = _RenderEngine(self)
        self._engine.start()  # builds visuals, installs the per-frame task, blocks

    # ---- API callable from update (synced via command queue) -----------
    def attach_user_navigator(self,
                              user_navigator: UserNavigator,
                              camera_object: Union[SceneObject, str, None] = None) -> None:
        """Attach a user navigator to a camera object.

        camera_object can be a SceneObject or a string name. Synced with the
        rendering thread; can be called from update().
        """
        if camera_object is not None:
            user_navigator.camera_object = camera_object
        self._navigators.append(user_navigator)

    def save_snapshot(self, view_name: str, image_path: Union[Path, str]) -> None:
        """Save a png image of a viewer to the given path.

        Synced with the rendering thread; can be called from update().
        """
        image_path = Path(image_path).resolve()
        if self._engine is not None:
            self._engine.enqueue(lambda: self._engine.save_snapshot(view_name, image_path))

    def enable_viewer(self, view_name: Optional[str], enable: bool) -> None:
        """Enable (show) or disable (hide) one or all viewers.

        If view_name is None it applies to all viewers. Synced with the rendering
        thread; can be called from update().
        """
        if view_name is not None:
            self.viewers[view_name].enable(enable)
        else:
            for v in self.viewers.values():
                v.enable(enable)


# =====================================================================
# Rendering engine (Panda3D). Owns the command queue and the render thread.
# =====================================================================
class _RenderEngine:
    """Wraps Panda3D's ShowBase and drives the per-frame loop.

    Everything in this class runs on the rendering thread. The only thread-safe
    entry point from other threads is :meth:`enqueue`, which appends a command to
    the queue (``collections.deque.append`` is atomic). Commands are drained at
    the start of every frame.
    """

    def __init__(self, owner: player):
        self.player = owner
        self.command_queue: "collections.deque[Callable[[], None]]" = collections.deque()
        self.base = None
        self._keys_down: Dict[str, bool] = {}
        self._scene_roots: Dict[int, object] = {}  # id(scene) -> NodePath
        self._last_time: float = 0.0

    # ---- thread-safe entry point ---------------------------------------
    def enqueue(self, command: Callable[[], None]) -> None:
        self.command_queue.append(command)

    def _drain(self) -> None:
        queue = self.command_queue
        while True:
            try:
                command = queue.popleft()
            except IndexError:
                break
            try:
                command()
            except Exception as exc:  # never let a bad command kill the renderer
                print(f"[grx] command failed: {exc}")

    # ---- lifecycle ------------------------------------------------------
    def start(self) -> None:
        from direct.showbase.ShowBase import ShowBase
        from panda3d.core import NodePath, PandaNode

        self.base = ShowBase()
        self.base.disableMouse()  # we drive the camera ourselves

        # wire the engine to every scene / viewer so their API can enqueue
        for viewer_obj in self.player.viewers.values():
            viewer_obj._engine = self
            scene = viewer_obj.scene
            scene._engine = self
            if id(scene) not in self._scene_roots:
                root = NodePath(PandaNode(f"{scene.name}_root"))
                root.reparentTo(self.base.render)
                self._scene_roots[id(scene)] = root
                scene._render_root = root
                # build visuals for everything already in the scene
                for obj in scene.objects():
                    self._build_object_visual(scene, obj)

        for viewer_obj in self.player.viewers.values():
            self.apply_background(viewer_obj)
            self.apply_grid(viewer_obj)

        self._install_input()
        self._last_time = 0.0
        self.base.taskMgr.add(self._frame_task, "grx_frame_task", sort=-50)
        self.base.run()

    def _frame_task(self, task):
        dt = task.time - self._last_time
        self._last_time = task.time
        if dt < 0.0:
            dt = 0.0

        # 1) apply queued visual commands
        self._drain()
        # 2) run user logic (may enqueue more commands)
        try:
            self.player._update()
        except NotImplementedError:
            pass
        except Exception as exc:
            print(f"[grx] player._update failed: {exc}")
        self._drain()
        # 3) run navigators
        for navigator in self.player._navigators:
            try:
                navigator.update(self, dt)
            except Exception as exc:
                print(f"[grx] navigator failed: {exc}")
        self._drain()

        self._reset_mouse_delta()
        return task.cont

    # ---- input ----------------------------------------------------------
    def _install_input(self):
        for key in ["w", "a", "s", "d", "q", "e"]:
            self.base.accept(key, self._set_key, [key, True])
            self.base.accept(f"{key}-up", self._set_key, [key, False])
        self._mouse_last = None
        self._mouse_delta = (0.0, 0.0)

    def _set_key(self, key: str, value: bool):
        self._keys_down[key] = value

    def is_key_down(self, key: str) -> bool:
        return self._keys_down.get(key, False)

    def mouse_delta(self):
        mw = self.base.mouseWatcherNode
        if mw is None or not mw.hasMouse():
            return (0.0, 0.0)
        x, y = mw.getMouseX(), mw.getMouseY()
        if self._mouse_last is None:
            self._mouse_last = (x, y)
            return (0.0, 0.0)
        dx = x - self._mouse_last[0]
        dy = y - self._mouse_last[1]
        self._mouse_last = (x, y)
        self._mouse_delta = (dx, dy)
        return self._mouse_delta

    def _reset_mouse_delta(self):
        self._mouse_delta = (0.0, 0.0)

    # ---- object / scene resolution -------------------------------------
    def resolve_object(self, obj: Union[SceneObject, str, None]) -> Optional[SceneObject]:
        if obj is None:
            return None
        if isinstance(obj, SceneObject):
            return obj
        for viewer_obj in self.player.viewers.values():
            try:
                return viewer_obj.scene.get_object(obj)
            except KeyError:
                continue
        return None

    # ---- visual commands (render thread) -------------------------------
    def create_visual(self, scene: Scene, obj: SceneObject) -> None:
        self._build_object_visual(scene, obj)

    def _build_object_visual(self, scene: Scene, obj: SceneObject) -> None:
        if obj._nodepath is not None:
            return
        nodepath = obj._build_nodepath()
        # parent under the parent object's node, or the scene root
        if obj._parent is not None and obj._parent._nodepath is not None:
            nodepath.reparentTo(obj._parent._nodepath)
        else:
            nodepath.reparentTo(scene._render_root)
        obj._nodepath = nodepath
        obj._apply_transform_to_node()
        # build children that may already exist
        for child in obj._children:
            self._build_object_visual(scene, child)

    def remove_visual(self, removed_objects: List[SceneObject]) -> None:
        for obj in removed_objects:
            if obj._nodepath is not None:
                obj._nodepath.removeNode()
                obj._nodepath = None

    def apply_background(self, viewer_obj: Viewer) -> None:
        if self.base is None:
            return
        self.base.setBackgroundColor(*viewer_obj.background_color)

    def apply_grid(self, viewer_obj: Viewer) -> None:
        scene = viewer_obj.scene
        if scene._render_root is None:
            return
        if viewer_obj._grid_np is not None:
            viewer_obj._grid_np.removeNode()
            viewer_obj._grid_np = None
        if viewer_obj.grid_visible:
            grid = _make_grid_nodepath("grid", viewer_obj.grid_size, viewer_obj.grid_color)
            grid.reparentTo(scene._render_root)
            viewer_obj._grid_np = grid

    def apply_enabled(self, viewer_obj: Viewer) -> None:
        scene = viewer_obj.scene
        if scene._render_root is None:
            return
        if viewer_obj._enabled:
            scene._render_root.show()
        else:
            scene._render_root.hide()

    def save_snapshot(self, view_name: str, image_path: Path) -> None:
        if self.base is None:
            return
        image_path.parent.mkdir(parents=True, exist_ok=True)
        self.base.win.saveScreenshot(str(image_path))


# =====================================================================
# Panda3D geometry builders (render thread only)
# =====================================================================
def _make_box_nodepath(name: str, size: List[float], color: Color):
    """Build a centered box NodePath. ``size`` is grx [width(x), height(y), depth(z)]."""
    from panda3d.core import (GeomVertexFormat, GeomVertexData, GeomVertexWriter,
                              Geom, GeomTriangles, GeomNode, NodePath)
    # grx -> panda: panda x = grx width, panda y = grx depth, panda z = grx height
    hx = size[0] / 2.0
    hz = size[1] / 2.0
    hy = size[2] / 2.0

    corners = [
        (-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),  # bottom
        (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz),      # top
    ]
    faces = [
        (0, 1, 2, 3, (0, 0, -1)),
        (4, 7, 6, 5, (0, 0, 1)),
        (0, 4, 5, 1, (0, -1, 0)),
        (2, 6, 7, 3, (0, 1, 0)),
        (1, 5, 6, 2, (1, 0, 0)),
        (0, 3, 7, 4, (-1, 0, 0)),
    ]

    fmt = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vdata.setNumRows(24)
    vwriter = GeomVertexWriter(vdata, "vertex")
    nwriter = GeomVertexWriter(vdata, "normal")
    cwriter = GeomVertexWriter(vdata, "color")

    tris = GeomTriangles(Geom.UHStatic)
    base_index = 0
    for a, b, c, d, normal in faces:
        for idx in (a, b, c, d):
            vwriter.addData3(*corners[idx])
            nwriter.addData3(*normal)
            cwriter.addData4(*color)
        tris.addVertices(base_index, base_index + 1, base_index + 2)
        tris.addVertices(base_index, base_index + 2, base_index + 3)
        base_index += 4

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    nodepath = NodePath(node)
    if color[3] < 1.0:
        from panda3d.core import TransparencyAttrib
        nodepath.setTransparency(TransparencyAttrib.MAlpha)
    return nodepath


def _make_lines_nodepath(name: str, segments, thickness: float = 1.0):
    """Build a NodePath of colored line segments.

    ``segments`` is an iterable of (p0, p1, color) where points are in panda coords.
    """
    from panda3d.core import LineSegs, NodePath
    lines = LineSegs(name)
    lines.setThickness(thickness)
    for p0, p1, color in segments:
        lines.setColor(*color)
        lines.moveTo(*p0)
        lines.drawTo(*p1)
    return NodePath(lines.create())


def _grx_point_to_panda(point):
    """Convert a grx (x,y,z) point to panda (x,z,y)."""
    return (point[0], point[2], point[1])


def _make_axes_nodepath(name: str, length: float, x_color, y_color, z_color):
    segments = [
        (_grx_point_to_panda((0, 0, 0)), _grx_point_to_panda((length, 0, 0)), x_color),
        (_grx_point_to_panda((0, 0, 0)), _grx_point_to_panda((0, length, 0)), y_color),
        (_grx_point_to_panda((0, 0, 0)), _grx_point_to_panda((0, 0, length)), z_color),
    ]
    return _make_lines_nodepath(name, segments, thickness=2.0)


def _make_arrow_nodepath(name: str, size: List[float], color: Color):
    length = size[2]
    head = min(length * 0.25, max(size[0], size[1]) * 2.0) or length * 0.25
    w = max(size[0], size[1]) * 0.5
    segments = [
        (_grx_point_to_panda((0, 0, 0)), _grx_point_to_panda((0, 0, length)), color),
        (_grx_point_to_panda((0, 0, length)), _grx_point_to_panda((w, 0, length - head)), color),
        (_grx_point_to_panda((0, 0, length)), _grx_point_to_panda((-w, 0, length - head)), color),
        (_grx_point_to_panda((0, 0, length)), _grx_point_to_panda((0, w, length - head)), color),
        (_grx_point_to_panda((0, 0, length)), _grx_point_to_panda((0, -w, length - head)), color),
    ]
    return _make_lines_nodepath(name, segments, thickness=2.0)


def _make_grid_nodepath(name: str, cell_size: float, color: Color, half_count: int = 20):
    """Build a grid on the grx floor (xz plane, y = 0)."""
    extent = cell_size * half_count
    segments = []
    for i in range(-half_count, half_count + 1):
        offset = i * cell_size
        # lines parallel to grx z (front)
        segments.append((_grx_point_to_panda((offset, 0, -extent)),
                         _grx_point_to_panda((offset, 0, extent)), color))
        # lines parallel to grx x (right)
        segments.append((_grx_point_to_panda((-extent, 0, offset)),
                         _grx_point_to_panda((extent, 0, offset)), color))
    return _make_lines_nodepath(name, segments, thickness=1.0)


__all__ = [
    "hello",
    "identity_transform", "as_transform", "translation_transform",
    "axis_angle_transform", "compose", "inverse_transform", "relative_transform",
    "SceneObject", "Cube", "Arrow", "arrow", "Axes", "axes",
    "Camera", "PinHoleCamera",
    "SceneLightObject", "SunLightObject",
    "Scene", "Viewer", "viewer",
    "UserNavigator", "ShooterUserNavigator",
    "player",
    "WHITE", "LIGHT_GRAY", "RED", "GREEN", "BLUE",
]
