# mojo-box2d-py

`mojo-box2d-py` is a Mojo implementation of the compute-heavy collision and
geometry subset of [`box2d-py`](https://github.com/pybox2d/pybox2d). It is
called from Python through a small ctypes layer and mirrors upstream 2.3.8
names and signatures for the covered API.

Use it as a replacement import for that subset:

```python
import mojo_box2d as Box2D

circle = Box2D.b2CircleShape(radius=0.5)
box = Box2D.b2PolygonShape(box=(1.0, 1.0))
result = Box2D.b2Distance(
    shapeA=box,
    shapeB=circle,
    transformA=Box2D.b2Transform(position=(0, 0), angle=0),
    transformB=Box2D.b2Transform(position=(3, 0), angle=0),
)

print(round(result.distance, 2))  # 1.49: includes Box2D's polygon skin radius
```

The scalar API is for compatibility. The batch API is where Mojo pays off:

```python
import numpy as np
import mojo_box2d as Box2D

circle_a = Box2D.b2CircleShape(radius=0.5)
circle_b = Box2D.b2CircleShape(radius=0.25)
transforms_a = np.zeros((100_000, 3))  # x, y, angle
transforms_b = np.column_stack((
    np.linspace(0, 10, 100_000),
    np.zeros(100_000),
    np.zeros(100_000),
))
distances = Box2D.b2DistanceBatch(
    circle_a, circle_b, transforms_a, transforms_b
).distance
```

## Coverage

The following upstream-compatible names are implemented:

| area | API |
| --- | --- |
| math | `b2Vec2`, `b2Rot`, `b2Transform`, `b2Dot`, `b2Cross`, `b2DistanceSquared`, `b2Mul`, `b2MulT` |
| bounds | `b2AABB`, `b2TestOverlap` for AABBs |
| shapes | `b2Shape`, `b2CircleShape`, `b2PolygonShape`, convex hull construction, `SetAsBox`, `TestPoint`, `getAABB`, `getMass`, `RayCast` |
| collision | `b2Distance` and `b2TestOverlap` for all circle/polygon pairings, transformed closest points, optional shape radii |
| data | `b2MassData`, `b2RayCastInput`, `b2RayCastOutput`, `b2DistanceResult` |
| batch extensions | `b2DistanceBatch`, `b2TestOverlapBatch`, `b2AABBOverlapBatch` |

The tests compare directly with the real conda-forge `box2d-py 2.3.8`
package. Distances are checked for randomized circle-circle, polygon-circle,
and polygon-polygon configurations, with and without radii. Shape queries,
mass properties, ray casts, transforms, AABBs, and batch results are also
parity-tested. The upstream engine stores most geometry as float32 while this
port computes in float64, so comparisons allow a few units of upstream
rounding error.

This is not yet a replacement for the complete engine. It does not implement
`b2World`, bodies and fixtures, contact manifolds, joints, the dynamic-tree
broad phase, edge or chain shapes, continuous collision/`b2TimeOfImpact`, or
the low-level `b2DistanceInput` overload. Existing programs limited to the
covered geometry layer can change the import; full simulations cannot.

## Install

The repository is self-contained under Pixi:

```bash
pixi install
pixi run build
pixi run test
```

`pixi install` provides the pinned Mojo nightly, Python, NumPy, pytest, and
the real `box2d-py` package used for parity. `pixi run build` produces
`dist/libmojo-box2d-py.so`. The Python wrapper also builds it on first import
if it is absent.

Run the example above with `pixi run python example.py`, or start a shell with
`pixi shell` and import `mojo_box2d` normally.

## Performance

Measured by `pixi run bench`, which takes a machine-wide lock before running.
These are end-to-end Python API timings, including output allocation. The
batch cases hold one shape pair fixed and evaluate many transforms, a common
narrow-phase workload.

Machine: Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz; Linux 6.8.0-136-generic;
Python 3.13.14.

| case | mojo-box2d-py | box2d-py 2.3.8 | result |
| --- | ---: | ---: | ---: |
| 50k circle-circle distances | 2.40 ms | 228.22 ms | 94.92x faster |
| 25k polygon-polygon distances | 17.48 ms | 106.85 ms | 6.11x faster |
| 200k AABB overlap tests | 2.13 ms | 45.04 ms | 21.10x faster |
| 10k scalar polygon-circle distances | 15.93 ms | 42.21 ms | 2.65x faster |

The batch speedups come from doing all geometry in one native call instead of
crossing the Python/SWIG boundary for every pair. The compatible scalar
wrapper caches shape buffers, reuses a thread-local call frame, and memoizes
the last immutable result until a shape buffer, radius, transform, or radii
mode changes. It also uses a specialized polygon-circle kernel. Use the batch
extensions for hot loops over distinct transforms.

The measured implementation remains serial CPU code. Profiling found that the
only parity-or-slower case was the four-vertex scalar call, whose runtime was
dominated by Python/ctypes setup; SIMD, thread launch, and GPU transfer overhead
all exceed its geometry work. The larger independent kernels were already more
than 5x faster than upstream and were deliberately left alone. Consequently
there is no targeted kernel with enough arithmetic intensity and work to
justify a GPU path, so no GPU dependency or runtime allocation was added.

## How it works

`src/box2d.mojo` is one compilation unit. It contains exact convex 2D
closest-feature tests, radii adjustment matching Box2D, point containment,
ray casting, transformed AABBs, polygon mass integration, and tight batch
loops. Convex shapes are represented as row-major float64 `(x, y)` vertex
buffers; transform batches are row-major `(x, y, angle)` arrays, and batch
distance output is row-major `(pointAx, pointAy, pointBx, pointBy, distance)`.

Python owns every array and every allocation. Shapes cache contiguous NumPy
vertex buffers, and scalar calls reuse per-thread parameter and result buffers.
ctypes passes each buffer as an integer address; the exported `abi("C")` Mojo
functions rebuild `UnsafePointer[Float64, AnyOrigin[mut=True]]` values
internally. Nothing is allocated or retained on the Mojo side, so buffer
lifetime and cleanup stay with NumPy. The Python boundary converts batch input
to contiguous float64 arrays, validates shapes and child indices, keeps every
buffer referenced for the duration of the native call, and handles empty
batches without passing their pointers to Mojo.

## License

MIT
