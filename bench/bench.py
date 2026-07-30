from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import Box2D as upstream
import mojo_box2d as mojo


def best_time(fn, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def up_xf(row):
    return upstream.b2Transform(
        upstream.b2Vec2(float(row[0]), float(row[1])),
        upstream.b2Rot(float(row[2])),
    )


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def cases():
    rng = np.random.default_rng(42)

    n = 50_000
    ta = np.zeros((n, 3))
    tb = np.c_[rng.uniform(-4, 4, n), rng.uniform(-4, 4, n), rng.uniform(-1, 1, n)]
    ma = mojo.b2CircleShape(radius=0.7, pos=(0.2, 0))
    mb = mojo.b2CircleShape(radius=0.4)
    ua = upstream.b2CircleShape(radius=0.7, pos=(0.2, 0))
    ub = upstream.b2CircleShape(radius=0.4)
    uta = [up_xf(row) for row in ta]
    utb = [up_xf(row) for row in tb]
    yield (
        "50k circle-circle distances",
        lambda: mojo.b2DistanceBatch(ma, mb, ta, tb),
        lambda: [
            upstream.b2Distance(
                shapeA=ua, shapeB=ub, transformA=a, transformB=b
            )
            for a, b in zip(uta, utb)
        ],
    )

    n = 25_000
    ta = np.zeros((n, 3))
    tb = np.c_[rng.uniform(-4, 4, n), rng.uniform(-3, 3, n), rng.uniform(-1, 1, n)]
    ma = mojo.b2PolygonShape(box=(1.0, 0.5))
    mb = mojo.b2PolygonShape(box=(0.6, 0.8))
    ua = upstream.b2PolygonShape(box=(1.0, 0.5))
    ub = upstream.b2PolygonShape(box=(0.6, 0.8))
    uta = [up_xf(row) for row in ta]
    utb = [up_xf(row) for row in tb]
    yield (
        "25k polygon-polygon distances",
        lambda: mojo.b2DistanceBatch(ma, mb, ta, tb),
        lambda: [
            upstream.b2Distance(
                shapeA=ua, shapeB=ub, transformA=a, transformB=b
            )
            for a, b in zip(uta, utb)
        ],
    )

    n = 200_000
    lower_a = rng.normal(size=(n, 2))
    lower_b = rng.normal(size=(n, 2))
    a = np.c_[lower_a, lower_a + rng.uniform(0.1, 1.5, (n, 2))]
    b = np.c_[lower_b, lower_b + rng.uniform(0.1, 1.5, (n, 2))]
    ua = [
        upstream.b2AABB(lowerBound=row[:2], upperBound=row[2:])
        for row in a
    ]
    ub = [
        upstream.b2AABB(lowerBound=row[:2], upperBound=row[2:])
        for row in b
    ]
    yield (
        "200k AABB overlap tests",
        lambda: mojo.b2AABBOverlapBatch(a, b),
        lambda: [upstream.b2TestOverlap(x, y) for x, y in zip(ua, ub)],
    )

    ma = mojo.b2PolygonShape(box=(1.0, 0.5))
    mb = mojo.b2CircleShape(radius=0.3)
    ua = upstream.b2PolygonShape(box=(1.0, 0.5))
    ub = upstream.b2CircleShape(radius=0.3)
    mt0 = mojo.b2Transform()
    mt = mojo.b2Transform((2, 0), mojo.b2Rot(0.2))
    ut0 = upstream.b2Transform(upstream.b2Vec2(0, 0), upstream.b2Rot(0.0))
    ut = upstream.b2Transform(upstream.b2Vec2(2, 0), upstream.b2Rot(0.2))
    yield (
        "10k scalar polygon-circle distances",
        lambda: [
            mojo.b2Distance(shapeA=ma, shapeB=mb, transformA=mt0, transformB=mt)
            for _ in range(10_000)
        ],
        lambda: [
            upstream.b2Distance(shapeA=ua, shapeB=ub, transformA=ut0, transformB=ut)
            for _ in range(10_000)
        ],
    )


def main():
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.release()}; Python {platform.python_version()}")
    print()
    print("| case | mojo-box2d-py | box2d-py 2.3.8 | result |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, theirs in cases():
        ours()
        theirs()
        a = best_time(ours, repeat=3)
        b = best_time(theirs, repeat=3)
        ratio = b / a
        result = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
        print(f"| {name} | {a * 1e3:.2f} ms | {b * 1e3:.2f} ms | {result} |")


if __name__ == "__main__":
    main()
