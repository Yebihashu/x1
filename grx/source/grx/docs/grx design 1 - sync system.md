THIS CODUMENT IS A CURRENT GRX SYSTEM DESIGN BASED ON SYNC RENDERING

# GRX lib - general
1. IMPORTANT: GRX is intended to be a real-time 3D engine, so it must be fast, efficient, and responsive.
2. GRX is currently based on the Panda3D rendering engine. Panda3D-specific code should be isolated inside the render backend layer, so the rendering backend can be changed later if needed.
3. The GRX coordinate system is defined in "\grx\source\grx\docs\coordinates system.md".
4. Units are meters and degrees.
5. Colors are RGBA values in the range 0.0-1.0. Example: [1.0, 0.0, 0.0, 1.0] is opaque red.
6. Transparency is supported. It can be enabled or disabled per object and also per viewer/scene through viewer settings.
7. GRX uses NumPy for efficient math calculations. The math backend is intentionally isolated so it can be replaced or accelerated later if needed.
8. Transforms are 4x4 matrices using column-vector convention. Transforming a homogeneous point is done as:
   transformed_point = transform_matrix @ vector_point
9. GRX uses a non-blocking log system. Log calls must not delay rendering, user input, or scene updates. Log calls enqueue messages quickly, and a separate logger thread handles printing/writing messages. The logger can print to stdout, file, or both. The default is stdout only.
   Note: the logger has its own queue. This is separate from render synchronization. GRX avoids a per-object render command queue for ordinary scene logic, but it does use a small render-thread command queue for Panda3D-side visual changes.
10. GRX should allow fluid navigation and rendering, even with two viewers or more, and should respond quickly to keyboard/mouse input.


# Sync Update/Render Scene Design
The following architecture describes the current implementation in `grx_lib.py`.


## Concept

GRX currently uses a **single render thread** model.

The thread that runs `player.run()` owns the Panda3D `ShowBase` instance, the window, the render loop, and all Panda3D objects. This same thread also runs the user callback `player._update()` and the user navigators every frame.

The logical scene state is updated immediately in Python objects, while Panda3D visual changes are applied on the render thread through a command queue. In other words:

- Python scene data is updated immediately.
- Panda3D state is updated later in the same frame, on the render thread.
- No other thread should touch Panda3D objects directly.

This keeps the system simple and responsive while still allowing the scene model to be queried immediately from Python.


## What is synchronous in the current design

The current GRX design is synchronous with respect to rendering:

- `player._update()` runs on the render thread.
- `UserNavigator.update()` runs on the render thread.
- `SceneObject.set_transform()` updates the Python scene state immediately and queues a Panda3D update command.
- `Viewer` state changes (background, grid, lighting, enabled flag) update Python state immediately and queue a Panda3D update command.
- `player.save_snapshot()` queues a render-thread snapshot command.

So the system is not a future async scene-packet design. Instead, it is a current sync render loop with a lightweight command queue for engine-side visual operations.


## Why the command queue exists

Panda3D objects must be touched only by the render thread. To avoid blocking and to keep `player._update()` usable from the same frame, GRX uses a queue of small callable commands.

Examples:

```python
scene.spawn_object(...)      -> queue create_visual(...)
scene.remove_object(...)     -> queue remove_visual(...)
obj.set_transform(...)       -> queue _apply_transform_to_node()
viewer.set_background_color() -> queue apply_background(...)
viewer.set_grid_visible(True) -> queue apply_grid(...)
```

The queue is drained inside the frame task, so queued changes are applied quickly and in order. This keeps the visual layer synchronized with the Python model without direct cross-thread Panda3D access.


## Frame flow in the current engine

The current render loop in `_RenderEngine._frame_task()` is:

```text
0. Pace: sleep until the scheduled wall-clock time for this tick
1. Drain queued visual commands
2. Run all user navigators (dt = system_tick_duration, fixed)
3. Drain queued visual commands again
4. Run player._update()
5. Drain queued visual commands again
6. Sync camera transforms from the Python scene to Panda3D cameras
7. Reset mouse delta
8. Increment tick count
9. Handle exit request if needed
```

This order is important:

- Pacing ensures the simulation runs at the configured wall-clock rate.
- Commands queued before the frame are applied first.
- Navigators can modify camera transforms early in the frame, using the fixed system tick as delta time.
- `player._update()` can freely modify scene objects and viewers while seeing the
  latest navigator effects.
- Camera NodePaths are synced at the end of the frame from the authoritative Python camera objects.
- The tick count increments after each completed tick, so `_update()` can read the current tick via `player.tick_count`.

The result is a predictable, deterministic render loop where the current Python scene state drives the visuals every frame.


## System Tick

GRX uses a **fixed logical system tick** that represents the progression of the simulation. The system tick is the authoritative time unit for the entire simulation and is independent of the actual wall-clock execution speed.

### Configuration

Two parameters are set at `player.__init__`:

- `system_tick_hz` (default 30): The logical simulation rate. `system_tick_duration = 1 / system_tick_hz` is the fixed delta time passed to navigators and used by all simulation logic.
- `run_tick_hz` (default: same as `system_tick_hz`; set to 0 for as-fast-as-possible): The wall-clock execution pacing rate. Controls how fast ticks execute in real time.

### Pacing

The engine paces execution using absolute scheduling:

```text
while running:
    scheduled_time = start_time + tick_count * run_tick_duration
    now = perf_counter()
    if now < scheduled_time:
        sleep(scheduled_time - now)
    run_one_simulation_tick()
    tick_count += 1
```

This prevents drift: if a tick takes longer than expected, the next tick starts immediately and the schedule catches up.

### Deterministic behavior

Every logical component of the simulation advances exactly one step per system tick:

- user navigators
- application update logic (`_update`)
- AI, physics, animations (when added)
- rendering

The simulation does not depend on the actual elapsed wall-clock time between iterations. Instead, all simulation logic progresses according to the configured system tick duration.

This provides:

- deterministic simulation behavior
- reproducible rendering results
- identical playback behavior
- the ability to execute the simulation slower or faster than real time without changing simulation results

The actual execution speed may be:

- real-time (`run_tick_hz` = `system_tick_hz`)
- slower than real-time (`run_tick_hz` < `system_tick_hz`, useful for debugging)
- faster than real-time (`run_tick_hz` > `system_tick_hz`, useful for offline generation)
- as fast as possible (`run_tick_hz` = 0)

while the logical simulation always advances by exactly one system tick per iteration.

### Tick count

The player exposes `tick_count`, which starts at 0 and increments by 1 after each completed tick. This can be read from `_update()` to determine which tick is currently executing.


## Scene model

### Scene

`Scene` is the authoritative registry of scene objects.

It stores:

- a dictionary of unique object names to `SceneObject`
- a list of root objects
- a recursive lock for safe scene mutations
- a reference to the active render engine when running

Scene responsibilities in the current design:

- create and remove objects
- keep the object registry consistent
- maintain parent/child relationships
- provide lookup by name
- enqueue render-side updates when needed

Names are always unique. If a requested name already exists, GRX appends an index to make it unique.

### SceneObject

`SceneObject` is the base class for all scene elements.

Each object stores:

- `name`
- `transform_local`
- parent pointer
- children list
- scene pointer
- visibility flag
- Panda3D `NodePath` reference, created later on the render thread

The important behavior is that the object keeps its local transform immediately in Python. World transform is computed by walking the parent chain.

Useful methods:

- `world_transform()`
- `get_transform(reference_object=None)`
- `set_transform(transform, reference_object=None)`
- `get_tranform()` and `set_tranform()` as backward-compatible aliases

`get_transform()` is immediate and non-blocking because it reads the Python scene model, not Panda3D state.

`set_transform()` updates the Python model immediately, then queues a render-thread command to apply the change to the NodePath if the scene is already running.

### Parenting

Objects can be spawned with a parent object or reparented by construction time.

The current design keeps local transforms relative to the parent. World transform is computed as:

```python
world = parent.world_transform() @ local_transform
```

When a subtree is removed, the entire subtree is collected and removed from the registry.


## Supported scene object types

The current `grx_lib.py` implementation provides these object types:

- `Pivot`: a transform-only parent node
- `Cube`: a colored cube mesh
- `Arrow`: an arrow pointing along local +z (front)
- `Axes`: a 3-axis gizmo with x/y/z colors
- `Camera`: base camera object
- `PinHoleCamera`: perspective camera parameters with focal length and principal point
- `OrthographicCamera`: orthographic camera with film size
- `SceneLightObject`: base class for scene lights
- `SunLightObject`: directional light

These are all scene objects, so they follow the same naming and parenting rules.


## Transform convention and math helpers

GRX uses 4x4 homogeneous transforms in column-vector convention.

The current code includes helper functions for:

- `identity_transform()`
- `as_transform()`
- `translation_transform()`
- `axis_angle_transform()`
- `compose()`
- `inverse_transform()`
- `relative_transform()`

The math layer also includes conversion between GRX transforms and Panda3D matrices:

- `_grx_to_panda_mat4()`
- `_panda_mat4_to_grx()`

This conversion is necessary because GRX uses its own coordinate system while Panda3D uses a different matrix convention.

The current code also defines the basis conversion between GRX and Panda3D by swapping the y and z axes.


## Coordinate system

The current GRX coordinate system is left-handed:

- x = right
- y = up
- z = front

Rotation directions are defined in the GRX math conventions and used consistently throughout the library.

This affects object orientation, camera navigation, arrow direction, axis gizmos, and Panda3D conversion helpers.


## Camera model

Cameras are regular scene objects.

The current implementation provides:

- `Camera`
- `PinHoleCamera`
- `OrthographicCamera`

### PinHoleCamera

A pinhole camera stores:

- `width`
- `height`
- `focal`
- `ppx`
- `ppy`

It can compute:

- a 3x3 projection matrix `K`
- the vertical field of view in degrees

### OrthographicCamera

An orthographic camera stores:

- `width`
- `height`
- `film_width`
- `film_height`

The render engine creates the appropriate Panda3D lens for each camera type.

Camera transforms are synced every frame from the Python camera object to the Panda3D camera NodePath.


## Viewer model

`Viewer` represents a render target for a scene.

A viewer stores:

- `scene`
- `camera`
- `headless` flag
- `enable_light` flag
- `background_color`
- `grid_size`
- `grid_color`
- `grid_visible`
- `_enabled`

A viewer can render either to a normal window or to an offscreen buffer.

Important viewer operations in the current design:

- `set_background_color(color)`
- `enable_lights(enable)`
- `set_grid(size, color=None)`
- `set_grid_visible(visible=True)`
- `enable(enable)`

These methods update the viewer object immediately and queue a render-thread command.

The current implementation also supports showing the same scene in multiple viewers with different viewer settings.


## Lights

Lights are scene objects too.

The current code supports a directional light object:

- `SunLightObject`

Lighting is controlled per viewer. That means two viewers can show the same scene with different lighting states.

The render engine applies lighting through the camera's initial render state, so lighting is viewer-specific rather than globally shared.

When lighting is disabled, the scene renders with flat object colors, which is useful for tests and exact image checks.


## Navigation and user input

`UserNavigator` is the base class for user-controlled camera logic.

The current implementation includes `ShooterUserNavigator`, which supports:

- mouse yaw and pitch
- `W/A/S/D` movement
- `Q/E` vertical movement
- configurable move rate, rotate rate, and mouse sensitivity

Navigators are updated on the render thread every frame.

This means navigation has immediate access to the latest input state and the current camera transform.

The current code stores key states in the render engine and samples mouse delta during the frame task.


## Images and snapshots

The current design includes a simple image capture path.

### `ImagesBuffer`

`ImagesBuffer` is a preallocated CPU buffer for storing multiple images.

It is useful when the application wants to keep rendered frames for later processing without reallocating memory every frame.

Supported operations:

- `image_reference(index)` returns a writable view to a slot
- `write_image(index, image)` copies image data into the slot
- `copy_to_cpu()` returns a full copy of the buffer

### `player.get_viewer_image()`

This reads the latest rendered image from a viewer into an `ImagesBuffer` slot.

In the current design, it must be called on the render thread, typically from inside `player._update()`.

### `player.save_snapshot()`

This queues a snapshot command for the render thread, which saves the current viewer output to disk.


## Logging

GRX uses a background-thread logger to avoid blocking rendering or input.

The current logger design:

- accepts log messages quickly
- places them on a queue
- prints from a dedicated logger thread
- can write to stdout, file, or both

This is intentionally separate from the render-thread command queue.


## Current engine lifecycle

### `player`

`player` is the base class for user applications.

Responsibilities:

- hold the viewers
- hold user navigators
- configure system tick rate (`system_tick_hz`) and execution pacing (`run_tick_hz`)
- implement `_update()` in subclasses
- start the engine via `run()`
- request exit via `exit()`

### `run()`

`player.run()` creates `_RenderEngine` and starts Panda3D's `ShowBase` loop.

### `_update()`

`_update()` is called once per system tick on the render thread.

The simulation advances by exactly `system_tick_duration` each tick. All logic should use this fixed step rather than wall-clock time. This is where the user typically changes the scene, viewer settings, or application logic.

### `exit()`

`exit()` asks the render loop to stop after the current frame.


## Important implementation notes

1. Scene data is updated immediately in Python.
2. Panda3D objects are created and modified only on the render thread.
3. Visual-affecting API calls enqueue lightweight commands and return immediately.
4. The current design does not use a separate async frame-packet system.
5. The current design does not require a background simulation thread.
6. Viewer and camera changes are applied per frame, so the display stays responsive.
7. The current design is intended to be simple enough for unit tests, interactive navigation, and small to medium real-time scenes.


## Summary

The current GRX sync system is built around these rules:

- one render thread drives Panda3D
- Python scene state is updated immediately
- Panda3D state is synchronized through a render-thread command queue
- `player._update()` and navigators run on the render thread
- simulation advances by a fixed system tick per frame, independent of wall-clock time
- execution pacing is configurable (real-time, slower, faster, or as-fast-as-possible)
- viewers, cameras, lights, and snapshots are controlled per frame
- logging stays non-blocking through a separate logger thread

This is the current practical design used by `grx_lib.py`.

