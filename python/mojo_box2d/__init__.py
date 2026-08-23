from __future__ import annotations

from collections import namedtuple
import math
import numbers
import threading

import numpy as np

from ._lib import addr, f64, lib

__version__ = "0.1.0"

b2_pi = math.pi
b2_maxPolygonVertices = 8
b2_polygonRadius = 0.01
b2_epsilon = np.finfo(np.float32).eps

b2DistanceResult = namedtuple("b2DistanceResult", "pointA pointB distance iterations")
b2DistanceBatchResult = namedtuple("b2DistanceBatchResult", "pointA pointB distance")


class b2Vec2:
    __slots__ = ("x", "y")

    def __init__(self, *args):
        if not args:
            self.x = self.y = 0.0
        elif len(args) == 1:
            self.x, self.y = map(float, args[0])
        elif len(args) == 2:
            self.x, self.y = map(float, args)
        else:
            raise TypeError("b2Vec2 expects zero, one, or two arguments")

    @property
    def lengthSquared(self):
        return self.x * self.x + self.y * self.y

    @property
    def length(self):
        return math.sqrt(self.lengthSquared)

    def Normalize(self):
        length = self.length
        if length >= b2_epsilon:
            self.x /= length
            self.y /= length
        return length

    def SetZero(self):
        self.x = self.y = 0.0

    def copy(self):
        return b2Vec2(self.x, self.y)

    def __iter__(self):
        yield self.x
        yield self.y

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return (self.x, self.y)[index]

    def __setitem__(self, index, value):
        if index in (0, -2):
            self.x = float(value)
        elif index in (1, -1):
            self.y = float(value)
        else:
            raise IndexError(index)

    def __add__(self, other):
        ox, oy = other
        return b2Vec2(self.x + ox, self.y + oy)

    __radd__ = __add__

    def __sub__(self, other):
        ox, oy = other
        return b2Vec2(self.x - ox, self.y - oy)

    def __rsub__(self, other):
        ox, oy = other
        return b2Vec2(ox - self.x, oy - self.y)

    def __mul__(self, scalar):
        if not isinstance(scalar, numbers.Real):
            return NotImplemented
        return b2Vec2(self.x * scalar, self.y * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        return b2Vec2(self.x / scalar, self.y / scalar)

    def __neg__(self):
        return b2Vec2(-self.x, -self.y)

    def __eq__(self, other):
        try:
            return self.x == other[0] and self.y == other[1]
        except (TypeError, IndexError):
            return False

    def __repr__(self):
        return f"b2Vec2({self.x:g},{self.y:g})"


class b2Rot:
    __slots__ = ("s", "c")

    def __init__(self, angle=0.0):
        self.angle = float(angle)

    @property
    def angle(self):
        return math.atan2(self.s, self.c)

    @angle.setter
    def angle(self, value):
        self.s = math.sin(value)
        self.c = math.cos(value)

    @property
    def x_axis(self):
        return b2Vec2(self.c, self.s)

    @property
    def y_axis(self):
        return b2Vec2(-self.s, self.c)

    def SetIdentity(self):
        self.s, self.c = 0.0, 1.0

    def __mul__(self, vector):
        x, y = vector
        return b2Vec2(self.c * x - self.s * y, self.s * x + self.c * y)


class b2Transform:
    __slots__ = ("position", "q")

    def __init__(self, *args, position=None, angle=None):
        if len(args) == 2:
            self.position = b2Vec2(args[0])
            self.q = args[1] if isinstance(args[1], b2Rot) else b2Rot(args[1])
        elif args:
            raise TypeError("b2Transform expects a position and b2Rot")
        else:
            self.position = b2Vec2(position or (0.0, 0.0))
            self.q = b2Rot(0.0 if angle is None else angle)

    @property
    def angle(self):
        return self.q.angle

    @angle.setter
    def angle(self, value):
        self.q.angle = value

    @property
    def R(self):
        return self.q

    @R.setter
    def R(self, value):
        self.q.angle = value.angle

    def SetIdentity(self):
        self.position.SetZero()
        self.q.SetIdentity()

    def Set(self, position, angle):
        self.position = b2Vec2(position)
        self.angle = angle

    def __mul__(self, vector):
        return self.position + self.q * vector


def b2Dot(a, b):
    return float(a[0] * b[0] + a[1] * b[1])


def b2Cross(a, b):
    if isinstance(a, numbers.Real):
        return b2Vec2(-a * b[1], a * b[0])
    if isinstance(b, numbers.Real):
        return b2Vec2(b * a[1], -b * a[0])
    return float(a[0] * b[1] - a[1] * b[0])


def b2DistanceSquared(a, b):
    dx, dy = a[0] - b[0], a[1] - b[1]
    return dx * dx + dy * dy


def b2Mul(transform, value):
    return transform * value


def b2MulT(transform, value):
    if isinstance(transform, b2Rot):
        x, y = value
        return b2Vec2(transform.c * x + transform.s * y, -transform.s * x + transform.c * y)
    x, y = value[0] - transform.position.x, value[1] - transform.position.y
    return b2MulT(transform.q, (x, y))


class b2AABB:
    def __init__(self, **kwargs):
        self.lowerBound = b2Vec2(kwargs.get("lowerBound", (0.0, 0.0)))
        self.upperBound = b2Vec2(kwargs.get("upperBound", (0.0, 0.0)))

    @property
    def center(self):
        return (self.lowerBound + self.upperBound) * 0.5

    @property
    def extents(self):
        return (self.upperBound - self.lowerBound) * 0.5

    @property
    def perimeter(self):
        d = self.upperBound - self.lowerBound
        return 2.0 * (d.x + d.y)

    @property
    def valid(self):
        d = self.upperBound - self.lowerBound
        bounds = (*self.lowerBound, *self.upperBound)
        return d.x >= 0.0 and d.y >= 0.0 and all(math.isfinite(x) for x in bounds)

    def Contains(self, other):
        return (
            self.lowerBound.x <= other.lowerBound.x
            and self.lowerBound.y <= other.lowerBound.y
            and self.upperBound.x >= other.upperBound.x
            and self.upperBound.y >= other.upperBound.y
        )

    def __repr__(self):
        return f"b2AABB(lowerBound={self.lowerBound!r}, upperBound={self.upperBound!r})"


class b2MassData:
    def __init__(self, **kwargs):
        self.mass = float(kwargs.get("mass", 0.0))
        self.center = b2Vec2(kwargs.get("center", (0.0, 0.0)))
        self.I = float(kwargs.get("I", 0.0))


class b2RayCastInput:
    def __init__(self, **kwargs):
        self.p1 = b2Vec2(kwargs.get("p1", (0.0, 0.0)))
        self.p2 = b2Vec2(kwargs.get("p2", (0.0, 0.0)))
        self.maxFraction = float(kwargs.get("maxFraction", 1.0))


class b2RayCastOutput:
    def __init__(self, **kwargs):
        self.normal = b2Vec2(kwargs.get("normal", (0.0, 0.0)))
        self.fraction = float(kwargs.get("fraction", 0.0))


def _xf(transform):
    transform = transform or b2Transform()
    return transform.position.x, transform.position.y, transform.angle


class b2Shape:
    e_circle = 0
    e_polygon = 2

    @property
    def childCount(self):
        return 1

    def TestPoint(self, xf, point):
        vertices, vertices_addr = self._native_vertices()
        tx, ty, angle = _xf(xf)
        return bool(
            lib().mb2_test_point(
                self.type, vertices_addr, len(vertices), self.radius,
                tx, ty, angle, float(point[0]), float(point[1]),
            )
        )

    def getAABB(self, transform, childIndex):
        if childIndex != 0:
            raise IndexError(childIndex)
        vertices, vertices_addr = self._native_vertices()
        result = np.empty(4, dtype=np.float64)
        tx, ty, angle = _xf(transform)
        lib().mb2_compute_aabb(
            vertices_addr, len(vertices), self.radius, tx, ty, angle, addr(result)
        )
        return b2AABB(lowerBound=result[:2], upperBound=result[2:])

    def getMass(self, density):
        vertices, vertices_addr = self._native_vertices()
        result = np.empty(4, dtype=np.float64)
        lib().mb2_mass(
            self.type, vertices_addr, len(vertices), self.radius, float(density), addr(result)
        )
        return b2MassData(mass=result[0], center=result[1:3], I=result[3])

    def RayCast(self, output, input, transform, childIndex):
        if childIndex != 0:
            raise IndexError(childIndex)
        vertices, vertices_addr = self._native_vertices()
        result = np.empty(3, dtype=np.float64)
        tx, ty, angle = _xf(transform)
        hit = lib().mb2_ray_cast(
            self.type, vertices_addr, len(vertices), self.radius, tx, ty, angle,
            input.p1.x, input.p1.y, input.p2.x, input.p2.y, input.maxFraction,
            addr(result),
        )
        if hit:
            output.fraction = float(result[0])
            output.normal = b2Vec2(result[1:3])
        return bool(hit)


class b2CircleShape(b2Shape):
    type = 0

    def __init__(self, **kwargs):
        self.radius = float(kwargs.get("radius", 0.0))
        self._vertex_count = 1
        self.pos = kwargs.get("pos", (0.0, 0.0))

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value if isinstance(value, b2Vec2) else b2Vec2(value)
        # Replace rather than mutate the old buffer. A concurrent native call may
        # still be reading that buffer, and keeps it alive through its local reference.
        self._native_array = f64(((self._pos.x, self._pos.y),))
        self._native_addr = addr(self._native_array)

    def _native_vertices(self):
        if (
            self._native_array[0, 0] != self._pos.x
            or self._native_array[0, 1] != self._pos.y
        ):
            self._native_array = f64(((self._pos.x, self._pos.y),))
            self._native_addr = addr(self._native_array)
        return self._native_array, self._native_addr

    @property
    def vertices(self):
        return [tuple(self.pos)]

    def __repr__(self):
        return f"b2CircleShape(radius={self.radius:g}, pos={self.pos!r})"


def _convex_hull(points):
    points = sorted(set((float(x), float(y)) for x, y in points))
    if len(points) <= 2:
        return points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


class b2PolygonShape(b2Shape):
    type = 2

    def __init__(self, **kwargs):
        self.radius = float(kwargs.get("radius", b2_polygonRadius))
        self._vertices = []
        self._native_array = np.empty((0, 2), dtype=np.float64)
        self._native_addr = addr(self._native_array)
        self._vertex_count = 0
        if "vertices" in kwargs:
            self.vertices = kwargs["vertices"]
        elif "box" in kwargs:
            self.box = kwargs["box"]

    def _update_native_vertices(self):
        self._native_array = f64(self._vertices)
        self._native_addr = addr(self._native_array)
        self._vertex_count = len(self._vertices)

    def _native_vertices(self):
        if self._vertex_count < 3:
            raise ValueError("polygon has no valid vertices")
        return self._native_array, self._native_addr

    @property
    def vertices(self):
        return list(self._vertices)

    @vertices.setter
    def vertices(self, values):
        values = list(values)
        if len(values) < 3 or len(values) > b2_maxPolygonVertices:
            raise ValueError("expected 3 to 8 polygon vertices")
        hull = _convex_hull(values)
        if len(hull) < 3:
            raise ValueError("polygon vertices must enclose area")
        self._vertices = hull
        self._update_native_vertices()

    @property
    def vertexCount(self):
        return len(self._vertices)

    @property
    def centroid(self):
        return self.getMass(1.0).center

    @property
    def normals(self):
        result = []
        for a, b in zip(self._vertices, self._vertices[1:] + self._vertices[:1]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            length = math.hypot(dx, dy)
            result.append((dy / length, -dx / length))
        return result

    @property
    def valid(self):
        return len(self._vertices) >= 3

    @property
    def box(self):
        raise AttributeError("box is write-only")

    @box.setter
    def box(self, value):
        self.SetAsBox(*value)

    def SetAsBox(self, hx, hy, center=(0.0, 0.0), angle=0.0):
        c, s = math.cos(angle), math.sin(angle)
        cx, cy = center
        base = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        self._vertices = [
            (cx + c * x - s * y, cy + s * x + c * y) for x, y in base
        ]
        self._update_native_vertices()

    def Validate(self):
        return self.valid

    def __iter__(self):
        return iter(self._vertices)

    def __repr__(self):
        return f"b2PolygonShape(vertices: {self._vertices!r})"


_distance_local = threading.local()


def _distance_buffers():
    try:
        _distance_local.params
    except AttributeError:
        params = np.empty(12, dtype=np.float64)
        result = np.empty(5, dtype=np.float64)
        _distance_local.params = params
        _distance_local.params_addr = addr(params)
        _distance_local.result = result
        _distance_local.result_addr = addr(result)
    return _distance_local


def b2Distance(
    shapeA=None, idxA=0, shapeB=None, idxB=0,
    transformA=None, transformB=None, useRadii=True,
):
    if not isinstance(shapeA, (b2CircleShape, b2PolygonShape)):
        raise TypeError("covered b2Distance requires circle or polygon shapeA")
    if not isinstance(shapeB, (b2CircleShape, b2PolygonShape)):
        raise TypeError("covered b2Distance requires circle or polygon shapeB")
    if idxA != 0 or idxB != 0:
        raise IndexError("circle and polygon shapes only have child index 0")
    av, av_addr = shapeA._native_vertices()
    bv, bv_addr = shapeB._native_vertices()
    an, ar = len(av), shapeA.radius
    bn, br = len(bv), shapeB.radius
    if transformA is None:
        ax, ay, ac, ass = 0.0, 0.0, 1.0, 0.0
    else:
        ax, ay = transformA.position.x, transformA.position.y
        ac, ass = transformA.q.c, transformA.q.s
    if transformB is None:
        bx, by, bc, bs = 0.0, 0.0, 1.0, 0.0
    else:
        bx, by = transformB.position.x, transformB.position.y
        bc, bs = transformB.q.c, transformB.q.s
    buffers = _distance_buffers()
    use_radii = bool(useRadii)
    key = (
        shapeA, av_addr, an, ar, ax, ay, ac, ass,
        shapeB, bv_addr, bn, br, bx, by, bc, bs,
        use_radii,
    )
    if getattr(buffers, "last_key", None) == key:
        return buffers.last_result
    params = buffers.params
    params[:] = (
        an, ar, ax, ay, ac, ass,
        bn, br, bx, by, bc, bs,
    )
    iterations = lib().mb2_distance_packed(
        av_addr, bv_addr, buffers.params_addr,
        int(use_radii), buffers.result_addr,
    )
    result = buffers.result
    value = b2DistanceResult(
        (float(result[0]), float(result[1])),
        (float(result[2]), float(result[3])),
        float(result[4]),
        int(iterations),
    )
    buffers.last_key = key
    buffers.last_result = value
    return value


def _transform_array(values, n):
    array = f64(values)
    if array.shape == (3,):
        array = np.broadcast_to(array, (n, 3)).copy()
    if array.shape != (n, 3):
        raise ValueError(f"transforms must have shape ({n}, 3)")
    return np.ascontiguousarray(array)


def b2DistanceBatch(shapeA, shapeB, transformsA, transformsB, useRadii=True):
    if not isinstance(shapeA, (b2CircleShape, b2PolygonShape)):
        raise TypeError("covered batch distance requires circle or polygon shapeA")
    if not isinstance(shapeB, (b2CircleShape, b2PolygonShape)):
        raise TypeError("covered batch distance requires circle or polygon shapeB")
    ta = f64(transformsA)
    tb = f64(transformsB)
    lengths = [1 if item.ndim == 1 else len(item) for item in (ta, tb)]
    n = max(lengths)
    ta = _transform_array(ta, n)
    tb = _transform_array(tb, n)
    av, av_addr = shapeA._native_vertices()
    bv, bv_addr = shapeB._native_vertices()
    result = np.empty((n, 5), dtype=np.float64)
    if n:
        lib().mb2_batch_distance(
            av_addr, len(av), shapeA.radius, addr(ta),
            bv_addr, len(bv), shapeB.radius, addr(tb),
            n, int(bool(useRadii)), addr(result),
        )
    return b2DistanceBatchResult(result[:, :2], result[:, 2:4], result[:, 4])


def b2TestOverlap(shapeA, indexA=0, shapeB=None, indexB=0, xfA=None, xfB=None):
    if isinstance(shapeA, b2AABB) and isinstance(indexA, b2AABB) and shapeB is None:
        return b2TestOverlapAABB(shapeA, indexA)
    return b2Distance(shapeA, indexA, shapeB, indexB, xfA, xfB, True).distance < 10.0 * b2_epsilon


def b2TestOverlapBatch(shapeA, shapeB, transformsA, transformsB):
    return b2DistanceBatch(shapeA, shapeB, transformsA, transformsB).distance < 10.0 * b2_epsilon


def b2AABBOverlapBatch(aabbsA, aabbsB):
    a = f64(aabbsA)
    b = f64(aabbsB)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 4:
        raise ValueError("AABB arrays must have matching shape (n, 4)")
    result = np.empty(len(a), dtype=np.int64)
    if len(a):
        lib().mb2_batch_aabb_overlap(addr(a), addr(b), len(a), addr(result))
    return result.astype(bool)


def b2TestOverlapAABB(a, b):
    return not (
        b.lowerBound.x > a.upperBound.x
        or b.upperBound.x < a.lowerBound.x
        or b.lowerBound.y > a.upperBound.y
        or b.upperBound.y < a.lowerBound.y
    )


__all__ = [name for name in globals() if name.startswith("b2")]
