import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

up = pytest.importorskip("Box2D")
import mojo_box2d as mb


def up_xf(x=0.0, y=0.0, angle=0.0):
    return up.b2Transform(up.b2Vec2(x, y), up.b2Rot(float(angle)))


def mb_xf(x=0.0, y=0.0, angle=0.0):
    return mb.b2Transform((x, y), mb.b2Rot(angle))


def assert_vec(actual, expected, atol=2e-6):
    assert np.allclose(tuple(actual), tuple(expected), atol=atol)


def shape_pairs():
    return [
        (
            up.b2CircleShape(radius=0.7, pos=(0.2, -0.1)),
            up.b2CircleShape(radius=0.4, pos=(-0.1, 0.2)),
            mb.b2CircleShape(radius=0.7, pos=(0.2, -0.1)),
            mb.b2CircleShape(radius=0.4, pos=(-0.1, 0.2)),
        ),
        (
            up.b2PolygonShape(box=(1.0, 0.6)),
            up.b2CircleShape(radius=0.4, pos=(-0.1, 0.2)),
            mb.b2PolygonShape(box=(1.0, 0.6)),
            mb.b2CircleShape(radius=0.4, pos=(-0.1, 0.2)),
        ),
        (
            up.b2PolygonShape(box=(1.0, 0.6)),
            up.b2PolygonShape(box=(0.5, 0.8)),
            mb.b2PolygonShape(box=(1.0, 0.6)),
            mb.b2PolygonShape(box=(0.5, 0.8)),
        ),
    ]


def test_vec2_construction_properties_and_mutation():
    ours = mb.b2Vec2(3, 4)
    theirs = up.b2Vec2(3, 4)
    assert ours.length == pytest.approx(theirs.length)
    assert ours.lengthSquared == pytest.approx(theirs.lengthSquared)
    assert ours.Normalize() == pytest.approx(theirs.Normalize())
    assert_vec(ours, theirs)
    ours.SetZero()
    theirs.SetZero()
    assert_vec(ours, theirs)


def test_vec2_arithmetic():
    ours = mb.b2Vec2(3, -4)
    theirs = up.b2Vec2(3, -4)
    for a, b in [
        (ours + (2, 5), theirs + (2, 5)),
        (ours - (2, 5), theirs - (2, 5)),
        (ours * 2.5, theirs * 2.5),
        (-ours, -theirs),
    ]:
        assert_vec(a, b)
    assert mb.b2Dot(ours, (2, 5)) == pytest.approx(up.b2Dot(theirs, (2, 5)))
    assert mb.b2Cross(ours, (2, 5)) == pytest.approx(up.b2Cross(theirs, (2, 5)))
    assert_vec(mb.b2Cross(2, ours), up.b2Cross(2, theirs))
    assert_vec(mb.b2Cross(ours, 2), up.b2Cross(theirs, 2))
    assert mb.b2DistanceSquared(ours, (1, 2)) == pytest.approx(
        up.b2DistanceSquared(theirs, (1, 2))
    )


def test_rotation_and_transform():
    ours_r, theirs_r = mb.b2Rot(0.37), up.b2Rot(0.37)
    assert_vec(ours_r * (2, -1), theirs_r * (2, -1))
    assert_vec(ours_r.x_axis, theirs_r.x_axis)
    ours = mb.b2Transform((3, -2), ours_r)
    theirs = up.b2Transform(up.b2Vec2(3, -2), theirs_r)
    assert_vec(ours * (2, -1), theirs * (2, -1))
    assert_vec(mb.b2Mul(ours, (2, -1)), up.b2Mul(theirs, (2, -1)))
    assert_vec(mb.b2MulT(ours, ours * (2, -1)), (2, -1))
    ours.SetIdentity()
    assert_vec(ours.position, (0, 0))
    assert ours.angle == pytest.approx(0)
    ours.Set((4, 5), 0.2)
    assert_vec(ours.position, (4, 5))


def test_aabb_properties_and_overlap():
    ours = mb.b2AABB(lowerBound=(1, 2), upperBound=(3, 5))
    theirs = up.b2AABB(lowerBound=(1, 2), upperBound=(3, 5))
    assert_vec(ours.center, theirs.center)
    assert_vec(ours.extents, theirs.extents)
    assert ours.perimeter == pytest.approx(theirs.perimeter)
    assert ours.valid == theirs.valid
    disjoint_o = mb.b2AABB(lowerBound=(4, 2), upperBound=(5, 3))
    disjoint_t = up.b2AABB(lowerBound=(4, 2), upperBound=(5, 3))
    assert mb.b2TestOverlap(ours, disjoint_o) == up.b2TestOverlap(theirs, disjoint_t)
    assert ours.Contains(mb.b2AABB(lowerBound=(1.5, 2.5), upperBound=(2, 4)))


@pytest.mark.parametrize("pair", shape_pairs())
@pytest.mark.parametrize("use_radii", [False, True])
def test_distance_randomized_parity(pair, use_radii):
    shape_a, shape_b, ours_a, ours_b = pair
    rng = np.random.default_rng(12)
    for x, y, angle in rng.uniform((-1.0, -1.0, -1.0), (4.0, 1.0, 1.0), (40, 3)):
        theirs = up.b2Distance(
            shapeA=shape_a,
            shapeB=shape_b,
            transformA=up_xf(0.1, -0.2, 0.17),
            transformB=up_xf(x, y, angle),
            useRadii=use_radii,
        )
        ours = mb.b2Distance(
            shapeA=ours_a,
            shapeB=ours_b,
            transformA=mb_xf(0.1, -0.2, 0.17),
            transformB=mb_xf(x, y, angle),
            useRadii=use_radii,
        )
        assert ours.distance == pytest.approx(theirs.distance, abs=3e-6)
        if theirs.distance > 1e-5:
            assert_vec(ours.pointA, theirs.pointA, atol=3e-6)
            assert_vec(ours.pointB, theirs.pointB, atol=3e-6)


@pytest.mark.parametrize("kind", ["circle", "polygon"])
def test_point_query(kind):
    if kind == "circle":
        theirs = up.b2CircleShape(radius=2, pos=(1, 3))
        ours = mb.b2CircleShape(radius=2, pos=(1, 3))
    else:
        theirs = up.b2PolygonShape(box=(2, 3))
        ours = mb.b2PolygonShape(box=(2, 3))
    tx_t, tx_o = up_xf(0.3, -0.5, 0.2), mb_xf(0.3, -0.5, 0.2)
    for point in [(-2, -2), (0, 0), (1, 3), (3, 3), (5, -1)]:
        assert ours.TestPoint(tx_o, point) == theirs.TestPoint(tx_t, point)


@pytest.mark.parametrize("kind", ["circle", "polygon"])
def test_compute_aabb(kind):
    if kind == "circle":
        theirs = up.b2CircleShape(radius=2, pos=(1, 3))
        ours = mb.b2CircleShape(radius=2, pos=(1, 3))
    else:
        theirs = up.b2PolygonShape(box=(2, 3))
        ours = mb.b2PolygonShape(box=(2, 3))
    actual = ours.getAABB(mb_xf(0.3, -0.5, 0.2), 0)
    expected = theirs.getAABB(up_xf(0.3, -0.5, 0.2), 0)
    assert_vec(actual.lowerBound, expected.lowerBound)
    assert_vec(actual.upperBound, expected.upperBound)


@pytest.mark.parametrize("kind", ["circle", "polygon"])
def test_mass_data(kind):
    if kind == "circle":
        theirs = up.b2CircleShape(radius=2, pos=(1, 3))
        ours = mb.b2CircleShape(radius=2, pos=(1, 3))
    else:
        theirs = up.b2PolygonShape(box=(2, 3))
        ours = mb.b2PolygonShape(box=(2, 3))
    actual, expected = ours.getMass(2.5), theirs.getMass(2.5)
    assert actual.mass == pytest.approx(expected.mass, rel=2e-6)
    assert actual.I == pytest.approx(expected.I, rel=2e-6)
    assert_vec(actual.center, expected.center)


@pytest.mark.parametrize("kind", ["circle", "polygon"])
def test_ray_cast(kind):
    if kind == "circle":
        theirs = up.b2CircleShape(radius=2, pos=(0, 0))
        ours = mb.b2CircleShape(radius=2, pos=(0, 0))
    else:
        theirs = up.b2PolygonShape(box=(2, 3))
        ours = mb.b2PolygonShape(box=(2, 3))
    input_t = up.b2RayCastInput(p1=(-10, 0), p2=(10, 0), maxFraction=1)
    input_o = mb.b2RayCastInput(p1=(-10, 0), p2=(10, 0), maxFraction=1)
    output_t, output_o = up.b2RayCastOutput(), mb.b2RayCastOutput()
    hit_t = theirs.RayCast(output_t, input_t, up_xf(0.3, -0.5, 0.2), 0)
    hit_o = ours.RayCast(output_o, input_o, mb_xf(0.3, -0.5, 0.2), 0)
    assert hit_o == hit_t
    assert output_o.fraction == pytest.approx(output_t.fraction, abs=2e-6)
    assert_vec(output_o.normal, output_t.normal)


def test_polygon_hull_and_oriented_box():
    points = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]
    ours = mb.b2PolygonShape(vertices=points)
    theirs = up.b2PolygonShape(vertices=points)
    assert ours.vertexCount == theirs.vertexCount
    assert np.allclose(sorted(ours.vertices), sorted(theirs.vertices))
    ours.SetAsBox(2, 1, (3, -1), 0.3)
    theirs.SetAsBox(2, 1, (3, -1), 0.3)
    assert np.allclose(ours.vertices, theirs.vertices, atol=2e-6)
    assert ours.Validate()
    assert_vec(ours.centroid, theirs.centroid)
    assert np.allclose(ours.normals, theirs.normals, atol=2e-6)
    assert ours.childCount == theirs.childCount


def test_batch_distance_matches_scalar_upstream():
    ours_a = mb.b2PolygonShape(box=(1, 0.5))
    ours_b = mb.b2CircleShape(radius=0.3)
    theirs_a = up.b2PolygonShape(box=(1, 0.5))
    theirs_b = up.b2CircleShape(radius=0.3)
    rng = np.random.default_rng(9)
    ta = rng.normal(size=(100, 3)) * (0.2, 0.2, 0.1)
    tb = rng.normal(size=(100, 3)) * (2.0, 1.0, 0.5)
    actual = mb.b2DistanceBatch(ours_a, ours_b, ta, tb)
    expected = np.array([
        up.b2Distance(
            shapeA=theirs_a, shapeB=theirs_b,
            transformA=up_xf(*a), transformB=up_xf(*b),
        ).distance
        for a, b in zip(ta, tb)
    ])
    assert np.allclose(actual.distance, expected, atol=3e-6)
    separated = expected > 1e-5
    expected_a = np.array([
        tuple(up.b2Distance(
            shapeA=theirs_a, shapeB=theirs_b,
            transformA=up_xf(*a), transformB=up_xf(*b),
        ).pointA)
        for a, b in zip(ta, tb)
    ])
    expected_b = np.array([
        tuple(up.b2Distance(
            shapeA=theirs_a, shapeB=theirs_b,
            transformA=up_xf(*a), transformB=up_xf(*b),
        ).pointB)
        for a, b in zip(ta, tb)
    ])
    assert np.allclose(actual.pointA[separated], expected_a[separated], atol=3e-6)
    assert np.allclose(actual.pointB[separated], expected_b[separated], atol=3e-6)


@pytest.mark.parametrize("vertex_count", [3, 5, 8])
@pytest.mark.parametrize("polygon_first", [False, True])
def test_scalar_polygon_circle_vertex_counts(vertex_count, polygon_first):
    vertices = [
        (math.cos(2 * math.pi * i / vertex_count), math.sin(2 * math.pi * i / vertex_count))
        for i in range(vertex_count)
    ]
    ours_polygon = mb.b2PolygonShape(vertices=vertices)
    ours_circle = mb.b2CircleShape(radius=0.23, pos=(0.1, -0.2))
    theirs_polygon = up.b2PolygonShape(vertices=vertices)
    theirs_circle = up.b2CircleShape(radius=0.23, pos=(0.1, -0.2))
    ours_shapes = (ours_polygon, ours_circle) if polygon_first else (ours_circle, ours_polygon)
    theirs_shapes = (
        (theirs_polygon, theirs_circle)
        if polygon_first else (theirs_circle, theirs_polygon)
    )
    for x, y, angle in [(1.7, -0.3, 0.2), (0.1, 0.0, -0.4), (-1.3, 0.7, 0.6)]:
        actual = mb.b2Distance(
            shapeA=ours_shapes[0], shapeB=ours_shapes[1],
            transformA=mb_xf(0.2, -0.1, 0.15), transformB=mb_xf(x, y, angle),
        )
        expected = up.b2Distance(
            shapeA=theirs_shapes[0], shapeB=theirs_shapes[1],
            transformA=up_xf(0.2, -0.1, 0.15), transformB=up_xf(x, y, angle),
        )
        assert actual.distance == pytest.approx(expected.distance, abs=3e-6)
        if expected.distance > 1e-5:
            assert_vec(actual.pointA, expected.pointA, atol=3e-6)
            assert_vec(actual.pointB, expected.pointB, atol=3e-6)


def test_scalar_distance_cached_buffers_track_mutation_and_threads():
    polygon = mb.b2PolygonShape(box=(1.0, 0.5))
    circle = mb.b2CircleShape(radius=0.2)
    transform = mb_xf(2.0, 0.0, 0.1)
    before = mb.b2Distance(
        shapeA=polygon, shapeB=circle, transformA=mb_xf(), transformB=transform
    ).distance
    circle.pos = (0.4, 0.0)
    polygon.SetAsBox(0.8, 0.4)
    after = mb.b2Distance(
        shapeA=polygon, shapeB=circle, transformA=mb_xf(), transformB=transform
    ).distance
    assert after != pytest.approx(before)
    circle.pos.x = 0.6
    assert mb.b2Distance(
        shapeA=polygon, shapeB=circle, transformA=mb_xf(), transformB=transform
    ).distance != pytest.approx(after)

    def evaluate(x):
        return mb.b2Distance(
            shapeA=polygon, shapeB=circle,
            transformA=mb_xf(), transformB=mb_xf(x, 0.0, 0.1),
        ).distance

    inputs = np.linspace(1.0, 3.0, 40)
    expected = [evaluate(x) for x in inputs]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(evaluate, inputs))
    assert np.allclose(actual, expected)


def test_batch_overlap_matches_scalar_upstream():
    rng = np.random.default_rng(4)
    lower_a = rng.normal(size=(500, 2))
    lower_b = rng.normal(size=(500, 2))
    a = np.c_[lower_a, lower_a + rng.uniform(0.1, 2, (500, 2))]
    b = np.c_[lower_b, lower_b + rng.uniform(0.1, 2, (500, 2))]
    actual = mb.b2AABBOverlapBatch(a, b)
    expected = [
        up.b2TestOverlap(
            up.b2AABB(lowerBound=aa[:2], upperBound=aa[2:]),
            up.b2AABB(lowerBound=bb[:2], upperBound=bb[2:]),
        )
        for aa, bb in zip(a, b)
    ]
    assert np.array_equal(actual, expected)


def test_overlap_shape_parity():
    ours_a, ours_b = mb.b2PolygonShape(box=(1, 1)), mb.b2CircleShape(radius=0.5)
    theirs_a, theirs_b = up.b2PolygonShape(box=(1, 1)), up.b2CircleShape(radius=0.5)
    for x in np.linspace(0, 3, 20):
        assert mb.b2TestOverlap(
            ours_a, 0, ours_b, 0, mb_xf(), mb_xf(x, 0)
        ) == up.b2TestOverlap(
            theirs_a, 0, theirs_b, 0, up_xf(), up_xf(x, 0)
        )


def test_batch_overlap_extension_and_empty_batches():
    a = mb.b2CircleShape(radius=0.5)
    b = mb.b2CircleShape(radius=0.5)
    ta = np.zeros((3, 3), dtype=np.float32)
    tb = np.array([[0, 0, 0], [2, 0, 0], [0.5, 0, 0]], dtype=np.float32)
    assert np.array_equal(mb.b2TestOverlapBatch(a, b, ta, tb), [True, False, True])
    empty_distance = mb.b2DistanceBatch(a, b, np.empty((0, 3)), np.empty((0, 3)))
    assert empty_distance.distance.shape == (0,)
    assert mb.b2AABBOverlapBatch(np.empty((0, 4)), np.empty((0, 4))).shape == (0,)


def test_data_objects_and_invalid_inputs_fail_before_native_call():
    mass = mb.b2MassData(mass=2, center=(3, 4), I=5)
    ray_input = mb.b2RayCastInput(p1=(1, 2), p2=(3, 4), maxFraction=0.5)
    ray_output = mb.b2RayCastOutput(normal=(0, 1), fraction=0.25)
    assert (mass.mass, tuple(mass.center), mass.I) == (2.0, (3.0, 4.0), 5.0)
    assert (tuple(ray_input.p1), tuple(ray_input.p2), ray_input.maxFraction) == (
        (1.0, 2.0), (3.0, 4.0), 0.5
    )
    assert (tuple(ray_output.normal), ray_output.fraction) == ((0.0, 1.0), 0.25)

    empty_polygon = mb.b2PolygonShape()
    with pytest.raises(ValueError, match="valid vertices"):
        empty_polygon.getAABB(mb.b2Transform(), 0)
    with pytest.raises(ValueError, match="valid vertices"):
        mb.b2Distance(
            shapeA=empty_polygon,
            shapeB=mb.b2CircleShape(radius=1),
        )
    with pytest.raises(ValueError, match=r"shape \(2, 3\)"):
        mb.b2DistanceBatch(
            mb.b2CircleShape(radius=1),
            mb.b2CircleShape(radius=1),
            np.zeros((2, 4)),
            np.zeros((2, 3)),
        )
