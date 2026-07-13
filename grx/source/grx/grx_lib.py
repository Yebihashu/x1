from __future__ import annotations
from typing import Callable, Dict, List, Optional, Union
from pathlib import Path
import itertools
import threading
import collections
import queue
import atexit
import math
import time
import numpy as np
from grx import grx_math
r"""
==================================================
Design - see design document: \x1\grx\source\grx\docs\grx design 1 - sync system.md
==================================================
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


# =====================================================================
# Logger (design item 11)
# ---------------------------------------------------------------------
# A non-blocking, threaded logger. Producers (any grx_lib object, possibly on
# the rendering thread) call ``log()`` which only pushes a message onto a
# thread-safe queue and returns immediately - it never does I/O, so it never
# stalls rendering or input handling. A dedicated background thread drains the
# queue and prints (with ANSI colors) to stdout, a file, or both.
# =====================================================================
class Logger:
    """A background-thread logger that never blocks the caller.

    Args:
        to_stdout : if True, messages are printed to stdout (default True).
        file_path : if given, messages are also appended to this file.
    """

    _COLORS = {
        "red": "\033[31m",
        "green": "\033[32m",
        "blue": "\033[34m",
        "yellow": "\033[33m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_blue": "\033[94m",
        "bright_yellow": "\033[93m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
    }
    _RESET = "\033[0m"

    def __init__(self, to_stdout: bool = True, file_path: Optional[Union[str, Path]] = None):
        self._queue: "queue.Queue" = queue.Queue()
        self._to_stdout = to_stdout
        self._file = open(file_path, "a", encoding="utf-8") if file_path else None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="grx-logger", daemon=True)
        self._thread.start()
        atexit.register(self.stop)

    def log(self, message: object, color: str = "white") -> None:
        """Queue a message for output. Non-blocking; returns immediately."""
        if self._closed:
            return
        self._queue.put((str(message), color))

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:  # sentinel -> stop
                    return
                message, color = item
                self._emit(message, color)
            finally:
                self._queue.task_done()

    def _emit(self, message: str, color: str) -> None:
        if self._to_stdout:
            code = self._COLORS.get(color, self._COLORS["white"])
            print(f"{code}{message}{self._RESET}")
        if self._file is not None:
            self._file.write(message + "\n")
            self._file.flush()

    def flush(self) -> None:
        """Block until all queued messages have been written."""
        self._queue.join()

    def configure(self, to_stdout: Optional[bool] = None,
                  file_path: Optional[Union[str, Path]] = None) -> None:
        """Reconfigure output sinks. Flushes pending messages first."""
        self.flush()
        if to_stdout is not None:
            self._to_stdout = to_stdout
        if file_path is not None:
            if self._file is not None:
                self._file.close()
            self._file = open(file_path, "a", encoding="utf-8")

    def stop(self) -> None:
        """Flush and stop the logger thread (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self.flush()
        self._queue.put(None)
        self._thread.join(timeout=1.0)
        if self._file is not None:
            self._file.close()
            self._file = None


# Default process-wide logger (stdout only, created lazily on first use).
_default_logger: Optional[Logger] = None
_logger_lock = threading.Lock()


def get_logger() -> Logger:
    """Return the process-wide default logger, creating it on first use."""
    global _default_logger
    if _default_logger is None:
        with _logger_lock:
            if _default_logger is None:
                _default_logger = Logger(to_stdout=True)
    return _default_logger


def set_logger(logger: Logger) -> None:
    """Replace the process-wide default logger."""
    global _default_logger
    _default_logger = logger


def log(message: object, color: str = "white") -> None:
    """Queue a message on the default logger. Non-blocking."""
    get_logger().log(message, color)


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

    Uses ``grx_math.rotation_vector_to_rotation_matrix``. ``axis`` need not be normalized.
    """
    v = np.asarray(axis, dtype=np.float64)[:3]
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return np.eye(4, dtype=np.float64)
    theta = math.radians(angle_deg)
    rotation_vector = (v / norm) * theta
    return grx_math.rotation_vector_to_rotation_matrix(rotation_vector)


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

def _panda_mat4_to_grx(transform: np.ndarray):
    """Convert a Panda3D matrix (row-vector convention) to grx column-major 4x4.

    Accepts Panda3D ``LMatrix4f``/``LMatrix4d`` or a numpy-like 4x4 row-major matrix.
    """
    if hasattr(transform, "getCell"):
        # Panda matrix type (LMatrix4f/LMatrix4d)
        row_major = np.array(
            [[transform.getCell(r, c) for c in range(4)] for r in range(4)],
            dtype=np.float64,
        )
    else:
        row_major = np.asarray(transform, dtype=np.float64)
        if row_major.shape == (16,):
            row_major = row_major.reshape((4, 4))
        if row_major.shape != (4, 4):
            raise ValueError(f"transform must be a 4x4 matrix, got shape {row_major.shape}")

    # Reverse of _grx_to_panda_mat4:
    #   row_major = panda.T
    #   panda = B @ grx @ B
    panda = row_major.T
    return _GRX_PANDA_BASIS @ panda @ _GRX_PANDA_BASIS


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

class Pivot(SceneObject):
    """A mesh-less scene object used purely as a transform/parent (pivot) node."""
    pass

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


class OrthographicCamera(Camera):
    """An orthographic camera.

    ``width`` and ``height`` are the rendered image size in pixels. ``film_width``
    and ``film_height`` are the orthographic view extents in meters (world units);
    when omitted they default to matching the pixel size at 100 px/meter.
    Looks along +z (front) with +y up, following the grx coordinate system.
    """

    def __init__(self,
                 name: Optional[str] = None,
                 transform: Optional[np.ndarray] = None,
                 parent_object: Optional[SceneObject] = None,
                 width: int = 600,
                 height: int = 400,
                 film_width: Optional[float] = None,
                 film_height: Optional[float] = None):
        super().__init__(name=name, transform=transform, parent_object=parent_object,
                         width=width, height=height)
        self.film_width = float(film_width) if film_width is not None else width / 100.0
        self.film_height = float(film_height) if film_height is not None else height / 100.0


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
                 camera: Union[Camera, str, None] = None,
                 headless: bool = False,
                 enable_light: bool = False):
        self.name = name
        self.scene = scene
        self.camera = camera
        # headless: render to an offscreen buffer instead of opening a window.
        self.headless = bool(headless)
        # lighting: default OFF so objects render with their own (flat) colors.
        # This is important for unit tests that verify rendered geometry, where
        # light-dependent shading would make exact color checks unreliable.
        self.enable_light = bool(enable_light)
        self.background_color: Color = list(WHITE)
        self.grid_size: float = 1.0
        self.grid_color: Color = list(LIGHT_GRAY)
        self.grid_visible: bool = False
        self._enabled: bool = True
        self._engine: Optional["_RenderEngine"] = None
        # Panda3D window / display objects, created by the engine.
        self._output = None       # GraphicsOutput (offscreen buffer or window)
        self._texture = None      # RAM-copy color Texture (headless)
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

    def enable_lights(self, enable: bool) -> None:
        """Enable or disable lights for this viewer.

        If enable is False, objects render with their own (flat) colors instead of
        light-computed colors. Applied per viewer, so two viewers of the same scene
        can independently render lit or unlit. Synced with the rendering thread;
        can be called from update().
        """
        self.enable_light = bool(enable)
        self._enqueue(lambda: self._engine.apply_lights(self))



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
# images buffer
# =====================================================================
class ImagesBuffer:
    """A pre-allocated buffer for a set of images.

    Purpose: fastest possible storage of frames coming from viewers at realtime,
    keeping them for later tasks without stalling the renderer. For example:

    * Example 1: analysis after playing. The application requests each frame (say
      100 frames), then closes the player and analyses the frames afterwards.
    * Example 2: realtime image processing, where processing can run over the
      buffer data, avoiding copies.

    The buffer is allocated once at construction. It can hand out a writable
    reference to a specific image slot, copy all images to a CPU numpy array, and
    report how many images it holds.

    Note:
        This implementation stores frames in a contiguous CPU numpy array, which
        is enough for the current needs and keeps the API stable. The storage
        backend can later be swapped for a GPU (e.g. torch/CUDA) buffer without
        changing this interface.
    """

    def __init__(self,
                 num_images: int,
                 width: int,
                 height: int,
                 channels: int = 3,
                 dtype=np.uint8):
        if num_images <= 0:
            raise ValueError("num_images must be positive")
        self._num_images = int(num_images)
        self._width = int(width)
        self._height = int(height)
        self._channels = int(channels)
        self._dtype = np.dtype(dtype)
        # shape: (N, H, W, C) - the natural layout for image data
        self._data = np.zeros((self._num_images, self._height, self._width, self._channels),
                              dtype=self._dtype)

    @property
    def num_images(self) -> int:
        return self._num_images

    @property
    def image_shape(self):
        """Return the (height, width, channels) of each image."""
        return (self._height, self._width, self._channels)

    def _check_index(self, index: int) -> bool:
        if index < 0 or index >= self._num_images:
            log(f"ImagesBuffer: image index {index} out of range [0, {self._num_images})", "red")
            return False
        return True

    def image_reference(self, index: int) -> Optional[np.ndarray]:
        """Return a writable view of the image slot at ``index`` (no copy).

        Returns None (and logs an error) if the index is out of range.
        """
        if not self._check_index(index):
            return None
        return self._data[index]

    def write_image(self, index: int, image: np.ndarray) -> bool:
        """Write ``image`` into the slot at ``index``. Returns False if invalid."""
        if not self._check_index(index):
            return False
        image = np.asarray(image)
        if image.shape != (self._height, self._width, self._channels):
            log(f"ImagesBuffer: image shape {image.shape} does not match "
                f"buffer shape {(self._height, self._width, self._channels)}", "red")
            return False
        self._data[index] = image
        return True

    def copy_to_cpu(self) -> np.ndarray:
        """Return a CPU numpy copy of all images, shape (N, H, W, C)."""
        return self._data.copy()

# =====================================================================
# Player
# =====================================================================
class player:
    """Base class for a player.

    The player updates the scene and renders it through one or more viewers. It
    attaches user navigators to camera objects, can save snapshots, and runs the
    scene. Subclasses implement ``_update`` (called once per system tick on the
    rendering thread).

    Args:
        viewers        : list of Viewer objects (or a single Viewer).
        system_tick_hz : logical simulation rate in Hz (default 30).
        run_tick_hz    : wall-clock pacing rate in Hz (default: same as
                         system_tick_hz; set to 0 for as-fast-as-possible).
    """

    def __init__(self, viewers: List[Viewer],
                 system_tick_hz: float = 30.0,
                 run_tick_hz: Optional[float] = None):
        if isinstance(viewers, Viewer):
            viewers = [viewers]
        self.viewers: Dict[str, Viewer] = {v.name: v for v in viewers}
        self._navigators: List[UserNavigator] = []
        self._engine: Optional["_RenderEngine"] = None
        if system_tick_hz <= 0:
            raise ValueError("system_tick_hz must be positive")
        self.system_tick_duration: float = 1.0 / system_tick_hz
        if run_tick_hz is None:
            self._run_tick_duration: float = self.system_tick_duration
        elif run_tick_hz <= 0:
            self._run_tick_duration = 0.0
        else:
            self._run_tick_duration = 1.0 / run_tick_hz
        self.tick_count: int = 0

    # ---- to be implemented by subclasses -------------------------------
    def _update(self) -> None:
        """Called once per system tick on the render thread.

        The simulation advances by exactly ``system_tick_duration`` each tick.
        All logic should use this fixed step rather than wall-clock time.
        Implemented by the inheriting class.
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

    def get_viewer_image(self, view_name: str, buffers: ImagesBuffer, image_index: int) -> bool:
        """Copy the latest rendered image of a viewer into ``buffers[image_index]``.

        Reads the viewer's rendered pixels, so it must be called on the rendering
        thread (i.e. from ``_update()``). Returns False (and logs) if the index is
        out of range or no rendered image is available yet.
        """
        if self._engine is None:
            log("get_viewer_image: engine is not running", "red")
            return False
        if view_name not in self.viewers:
            log(f"get_viewer_image: no viewer named {view_name!r}", "red")
            return False
        if image_index < 0 or image_index >= buffers.num_images:
            log(f"get_viewer_image: image index {image_index} out of range", "red")
            return False

        viewer_obj = self.viewers[view_name]
        image = self._engine.read_viewer_image(viewer_obj)
        if image is None:
            log(f"get_viewer_image: viewer {view_name!r} has no rendered image yet", "red")
            return False
        return buffers.write_image(image_index, image)

    def exit(self) -> None:
        """Stop the player: drain the command queue and end the rendering loop.

        Safe to call from ``_update()``. The remaining queued commands are applied,
        background threads (logger) are flushed, and ``run()`` returns.
        """
        if self._engine is not None:
            self._engine.request_exit()


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
        self._exit_requested: bool = False
        self._start_time: Optional[float] = None

    # ---- thread-safe entry point ---------------------------------------
    def enqueue(self, command: Callable[[], None]) -> None:
        self.command_queue.append(command)

    def request_exit(self) -> None:
        """Ask the render loop to stop after the current frame (thread-safe)."""
        self._exit_requested = True

    def _drain(self) -> None:
        command_queue = self.command_queue
        while True:
            try:
                command = command_queue.popleft()
            except IndexError:
                break
            try:
                command()
            except Exception as exc:  # never let a bad command kill the renderer
                log(f"[grx] command failed: {exc}", "red")

    # ---- lifecycle ------------------------------------------------------
    def start(self) -> None:
        from panda3d.core import loadPrcFileData

        viewers = list(self.player.viewers.values())
        all_headless = bool(viewers) and all(v.headless for v in viewers)
        if all_headless:
            # no on-screen window: render into offscreen buffers only.
            loadPrcFileData("grx-headless", "window-type offscreen")
            loadPrcFileData("grx-headless", "audio-library-name null")

        from direct.showbase.ShowBase import ShowBase
        from panda3d.core import NodePath, PandaNode

        self.base = ShowBase()
        try:
            self.base.disableMouse()  # we drive the camera ourselves
        except Exception:
            pass

        # wire the engine to every scene / viewer, and build scene roots.
        for viewer_obj in viewers:
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

        # per-viewer render targets (offscreen buffer or window) + cameras.
        for viewer_obj in viewers:
            self._setup_viewer(viewer_obj)

        self._install_input()
        self.base.taskMgr.add(self._frame_task, "grx_frame_task", sort=-50)
        log("[grx] engine started", "bright_green")
        self.base.run()
        # base.run() returns only after the loop is stopped (request_exit).
        self._teardown()

    def _frame_task(self, task):
        if self._start_time is None:
            self._start_time = time.perf_counter()

        if self.player._run_tick_duration > 0:
            scheduled = self._start_time + self.player.tick_count * self.player._run_tick_duration
            now = time.perf_counter()
            if now < scheduled:
                time.sleep(scheduled - now)

        dt = self.player.system_tick_duration

        # 1) apply queued visual commands
        self._drain()
        # 2) run navigators first (their effects should be visible to _update)
        for navigator in self.player._navigators:
            try:
                navigator.update(self, dt)
            except Exception as exc:
                log(f"[grx] navigator failed: {exc}", "red")
        self._drain()
        # 3) run user logic (may enqueue more commands)
        try:
            self.player._update()
        except NotImplementedError:
            pass
        except Exception as exc:
            log(f"[grx] player._update failed: {exc}", "red")
        self._drain()
        # 4) sync each viewer's camera transform from the data model
        for viewer_obj in self.player.viewers.values():
            self._update_viewer_camera_transform(viewer_obj)

        self._reset_mouse_delta()
        self.player.tick_count += 1

        # 5) honor an exit request: drain once more, then stop the loop.
        if self._exit_requested:
            self._drain()
            self.base.taskMgr.stop()
            return task.done
        return task.cont

    def _teardown(self) -> None:
        """Clean up Panda3D resources and flush the logger (after the loop)."""
        try:
            get_logger().flush()
        except Exception:
            pass
        try:
            if self.base is not None:
                self.base.destroy()
        except Exception:
            pass
        log("[grx] engine stopped", "bright_green")

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

    def _resolve_camera(self, viewer_obj: Viewer) -> Optional[Camera]:
        camera = viewer_obj.camera
        if camera is None:
            return None
        if isinstance(camera, Camera):
            return camera
        try:
            resolved = viewer_obj.scene.get_object(camera)
        except KeyError:
            return None
        return resolved if isinstance(resolved, Camera) else None

    # ---- per-viewer render target setup (render thread) ----------------
    def _setup_viewer(self, viewer_obj: Viewer) -> None:
        from panda3d.core import Texture

        camera = self._resolve_camera(viewer_obj)
        width = camera.width if camera is not None else 640
        height = camera.height if camera is not None else 480
        scene_root = self._scene_roots[id(viewer_obj.scene)]

        if viewer_obj.headless:
            texture = Texture(f"{viewer_obj.name}_tex")
            buffer = self.base.win.makeTextureBuffer(f"{viewer_obj.name}_buffer",
                                                     width, height, texture, True)
            buffer.setSort(-100)
            viewer_obj._output = buffer
            viewer_obj._texture = texture
            camera_np = self.base.makeCamera(buffer, scene=scene_root)
        else:
            viewer_obj._output = self.base.win
            viewer_obj._texture = None
            camera_np = self.base.makeCamera(self.base.win, scene=scene_root)

        # position the panda camera in world space (not under base.camera)
        camera_np.reparentTo(self.base.render)
        viewer_obj._camera_np = camera_np

        self._apply_lens(viewer_obj, camera, camera_np)
        self.apply_background(viewer_obj)
        self.apply_grid(viewer_obj)
        self.apply_lights(viewer_obj)
        self._update_viewer_camera_transform(viewer_obj)

    def _apply_lens(self, viewer_obj: Viewer, camera: Optional[Camera], camera_np) -> None:
        from panda3d.core import OrthographicLens, PerspectiveLens

        if isinstance(camera, OrthographicCamera):
            lens = OrthographicLens()
            lens.setFilmSize(camera.film_width, camera.film_height)
            # generous depth range so objects far from the (orthographic) camera
            # are not clipped.
            lens.setNearFar(-100000.0, 100000.0)
            camera_np.node().setLens(lens)
        elif isinstance(camera, PinHoleCamera):
            lens = PerspectiveLens()
            lens.setAspectRatio(camera.width / float(camera.height))
            lens.setFov(camera.vertical_fov_deg() * camera.width / float(camera.height),
                        camera.vertical_fov_deg())
            camera_np.node().setLens(lens)
        # otherwise keep Panda's default lens

    def _update_viewer_camera_transform(self, viewer_obj: Viewer) -> None:
        if viewer_obj._camera_np is None:
            return
        camera = self._resolve_camera(viewer_obj)
        if camera is None:
            return
        viewer_obj._camera_np.setMat(_grx_to_panda_mat4(camera.world_transform()))

    def read_viewer_image(self, viewer_obj: Viewer) -> Optional[np.ndarray]:
        """Read the viewer's latest rendered pixels as an (H, W, 3) uint8 array."""
        texture = viewer_obj._texture
        if texture is None or not texture.hasRamImage():
            return None
        xsize = texture.getXSize()
        ysize = texture.getYSize()
        ram = texture.getRamImageAs("RGB")
        if ram is None:
            return None
        raw = ram.getData() if hasattr(ram, "getData") else bytes(ram)
        array = np.frombuffer(raw, dtype=np.uint8)
        if array.size != xsize * ysize * 3:
            return None
        array = array.reshape(ysize, xsize, 3)
        # Panda stores images bottom-to-top; flip to top-to-bottom.
        return np.ascontiguousarray(array[::-1])

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
        from panda3d.core import LColor
        color = LColor(*viewer_obj.background_color)
        output = viewer_obj._output
        if output is not None:
            output.setClearColor(color)
            output.setClearColorActive(True)
        elif self.base is not None:
            self.base.setBackgroundColor(color)

    def apply_lights(self, viewer_obj: Viewer) -> None:
        """Enable or disable lighting for a viewer via its camera's initial state.

        Applying the light state on the camera (not the shared scene root) keeps
        lighting per-viewer, so viewers of the same scene can differ. When lighting
        is off, all lights are forced off and geometry shows its own vertex colors.
        """
        from panda3d.core import LightAttrib, RenderState

        camera_np = viewer_obj._camera_np
        if camera_np is None:
            return
        camera_node = camera_np.node()

        if viewer_obj.enable_light:
            light_attrib = LightAttrib.make()
            for obj in viewer_obj.scene.objects():
                if isinstance(obj, SceneLightObject) and obj._nodepath is not None:
                    light_attrib = light_attrib.addOnLight(obj._nodepath)
            camera_node.setInitialState(RenderState.make(light_attrib))
        else:
            camera_node.setInitialState(RenderState.make(LightAttrib.makeAllOff()))

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
        if self.base is None or view_name not in self.player.viewers:
            return
        viewer_obj = self.player.viewers[view_name]
        output = viewer_obj._output
        if output is None:
            return
        image_path.parent.mkdir(parents=True, exist_ok=True)
        output.saveScreenshot(str(image_path))


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
    "Logger", "get_logger", "set_logger", "log",
    "identity_transform", "as_transform", "translation_transform",
    "axis_angle_transform", "compose", "inverse_transform", "relative_transform",
    "SceneObject", "Cube", "Arrow", "arrow", "Axes", "axes",
    "Camera", "PinHoleCamera", "OrthographicCamera",
    "SceneLightObject", "SunLightObject",
    "Scene", "Viewer",
    "UserNavigator", "ShooterUserNavigator",
    "ImagesBuffer", "player",
    "WHITE", "LIGHT_GRAY", "RED", "GREEN", "BLUE",
]
