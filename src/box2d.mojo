from std.math import cos, sin, sqrt


comptime Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]


def px(v: Ptr, i: Int, tx: Float64, c: Float64, s: Float64) -> Float64:
    return tx + c * v[2 * i] - s * v[2 * i + 1]


def py(v: Ptr, i: Int, ty: Float64, c: Float64, s: Float64) -> Float64:
    return ty + s * v[2 * i] + c * v[2 * i + 1]


def point_in_polygon(
    v: Ptr, n: Int, tx: Float64, ty: Float64, c: Float64, s: Float64,
    qx: Float64, qy: Float64,
) -> Bool:
    if n < 3:
        return False
    var sign = 0
    for i in range(n):
        var j = (i + 1) % n
        var ax = px(v, i, tx, c, s)
        var ay = py(v, i, ty, c, s)
        var bx = px(v, j, tx, c, s)
        var by = py(v, j, ty, c, s)
        var cross = (bx - ax) * (qy - ay) - (by - ay) * (qx - ax)
        if abs(cross) <= 1.0e-12:
            continue
        var current = 1 if cross > 0.0 else -1
        if sign == 0:
            sign = current
        elif sign != current:
            return False
    return True


def segment_intersects(
    ax: Float64, ay: Float64, bx: Float64, by: Float64,
    cx: Float64, cy: Float64, dx: Float64, dy: Float64,
) -> Bool:
    var rx = bx - ax
    var ry = by - ay
    var sx = dx - cx
    var sy = dy - cy
    var den = rx * sy - ry * sx
    if abs(den) <= 1.0e-14:
        return False
    var qx = cx - ax
    var qy = cy - ay
    var t = (qx * sy - qy * sx) / den
    var u = (qx * ry - qy * rx) / den
    return t >= 0.0 and t <= 1.0 and u >= 0.0 and u <= 1.0


def consider_point_segment(
    qx: Float64, qy: Float64,
    ax: Float64, ay: Float64, bx: Float64, by: Float64,
    q_is_a: Bool, best: Ptr,
):
    var ex = bx - ax
    var ey = by - ay
    var denom = ex * ex + ey * ey
    var t = 0.0
    if denom > 0.0:
        t = ((qx - ax) * ex + (qy - ay) * ey) / denom
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
    var hx = ax + t * ex
    var hy = ay + t * ey
    var dx = qx - hx
    var dy = qy - hy
    var d2 = dx * dx + dy * dy
    if d2 < best[4]:
        best[4] = d2
        if q_is_a:
            best[0] = qx
            best[1] = qy
            best[2] = hx
            best[3] = hy
        else:
            best[0] = hx
            best[1] = hy
            best[2] = qx
            best[3] = qy


def raw_polygon_circle_distance(
    polygon: Ptr, n: Int,
    ptx: Float64, pty: Float64, pc: Float64, ps: Float64,
    circle: Ptr,
    ctx: Float64, cty: Float64, cc: Float64, cs: Float64,
    polygon_is_a: Bool, best: Ptr,
) -> Int:
    var circle_x = px(circle, 0, ctx, cc, cs)
    var circle_y = py(circle, 0, cty, cc, cs)
    var dx = circle_x - ptx
    var dy = circle_y - pty
    var qx = pc * dx + ps * dy
    var qy = -ps * dx + pc * dy
    var best_d2 = 1.7976931348623157e308
    var best_x = 0.0
    var best_y = 0.0
    var inside = True
    for i in range(n):
        var j = (i + 1) % n
        var ax = polygon[2 * i]
        var ay = polygon[2 * i + 1]
        var bx = polygon[2 * j]
        var by = polygon[2 * j + 1]
        var ex = bx - ax
        var ey = by - ay
        if ex * (qy - ay) - ey * (qx - ax) < -1.0e-12:
            inside = False
        var denom = ex * ex + ey * ey
        var t = 0.0
        if denom > 0.0:
            t = ((qx - ax) * ex + (qy - ay) * ey) / denom
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0
        var hx = ax + t * ex
        var hy = ay + t * ey
        var ddx = qx - hx
        var ddy = qy - hy
        var d2 = ddx * ddx + ddy * ddy
        if d2 < best_d2:
            best_d2 = d2
            best_x = hx
            best_y = hy

    if inside:
        best_x = qx
        best_y = qy
        best_d2 = 0.0
    var hit_x = ptx + pc * best_x - ps * best_y
    var hit_y = pty + ps * best_x + pc * best_y
    if polygon_is_a:
        best[0] = hit_x
        best[1] = hit_y
        best[2] = circle_x
        best[3] = circle_y
    else:
        best[0] = circle_x
        best[1] = circle_y
        best[2] = hit_x
        best[3] = hit_y
    best[4] = best_d2
    return n + 1


def raw_distance(
    a: Ptr, na: Int, atx: Float64, aty: Float64, ac: Float64, ass: Float64,
    b: Ptr, nb: Int, btx: Float64, bty: Float64, bc: Float64, bs: Float64,
    best: Ptr,
) -> Int:
    if na >= 3 and nb == 1:
        return raw_polygon_circle_distance(
            a, na, atx, aty, ac, ass,
            b, btx, bty, bc, bs, True, best,
        )
    if na == 1 and nb >= 3:
        return raw_polygon_circle_distance(
            b, nb, btx, bty, bc, bs,
            a, atx, aty, ac, ass, False, best,
        )
    best[4] = 1.7976931348623157e308
    var a0x = px(a, 0, atx, ac, ass)
    var a0y = py(a, 0, aty, ac, ass)
    var b0x = px(b, 0, btx, bc, bs)
    var b0y = py(b, 0, bty, bc, bs)

    if point_in_polygon(a, na, atx, aty, ac, ass, b0x, b0y):
        best[0] = b0x
        best[1] = b0y
        best[2] = b0x
        best[3] = b0y
        best[4] = 0.0
        return 1
    if point_in_polygon(b, nb, btx, bty, bc, bs, a0x, a0y):
        best[0] = a0x
        best[1] = a0y
        best[2] = a0x
        best[3] = a0y
        best[4] = 0.0
        return 1

    if na >= 2 and nb >= 2:
        var ae = na if na >= 3 else 1
        var be = nb if nb >= 3 else 1
        for i in range(ae):
            var ai = (i + 1) % na
            var ax = px(a, i, atx, ac, ass)
            var ay = py(a, i, aty, ac, ass)
            var ax2 = px(a, ai, atx, ac, ass)
            var ay2 = py(a, ai, aty, ac, ass)
            for j in range(be):
                var bj = (j + 1) % nb
                var bx = px(b, j, btx, bc, bs)
                var by = py(b, j, bty, bc, bs)
                var bx2 = px(b, bj, btx, bc, bs)
                var by2 = py(b, bj, bty, bc, bs)
                if segment_intersects(ax, ay, ax2, ay2, bx, by, bx2, by2):
                    best[0] = ax
                    best[1] = ay
                    best[2] = ax
                    best[3] = ay
                    best[4] = 0.0
                    return 1

    if nb >= 2:
        var be = nb if nb >= 3 else 1
        for i in range(na):
            var qx = px(a, i, atx, ac, ass)
            var qy = py(a, i, aty, ac, ass)
            for j in range(be):
                var bj = (j + 1) % nb
                consider_point_segment(
                    qx, qy,
                    px(b, j, btx, bc, bs), py(b, j, bty, bc, bs),
                    px(b, bj, btx, bc, bs), py(b, bj, bty, bc, bs),
                    True, best,
                )
    if na >= 2:
        var ae = na if na >= 3 else 1
        for i in range(nb):
            var qx = px(b, i, btx, bc, bs)
            var qy = py(b, i, bty, bc, bs)
            for j in range(ae):
                var aj = (j + 1) % na
                consider_point_segment(
                    qx, qy,
                    px(a, j, atx, ac, ass), py(a, j, aty, ac, ass),
                    px(a, aj, atx, ac, ass), py(a, aj, aty, ac, ass),
                    False, best,
                )
    if na == 1 and nb == 1:
        var dx = b0x - a0x
        var dy = b0y - a0y
        best[0] = a0x
        best[1] = a0y
        best[2] = b0x
        best[3] = b0y
        best[4] = dx * dx + dy * dy
    return na + nb


def apply_radii(
    ra: Float64, rb: Float64, use_radii: Bool, result: Ptr,
):
    var raw = sqrt(result[4])
    if use_radii:
        var total_radius = ra + rb
        if raw > total_radius and raw > 1.0e-14:
            var nx = (result[2] - result[0]) / raw
            var ny = (result[3] - result[1]) / raw
            result[0] += ra * nx
            result[1] += ra * ny
            result[2] -= rb * nx
            result[3] -= rb * ny
            result[4] = raw - total_radius
        else:
            var mx = 0.5 * (result[0] + result[2])
            var my = 0.5 * (result[1] + result[3])
            result[0] = mx
            result[1] = my
            result[2] = mx
            result[3] = my
            result[4] = 0.0
    else:
        result[4] = raw


def distance(
    a: Ptr, na: Int, ra: Float64,
    atx: Float64, aty: Float64, ac: Float64, ass: Float64,
    b: Ptr, nb: Int, rb: Float64,
    btx: Float64, bty: Float64, bc: Float64, bs: Float64,
    use_radii: Bool, result: Ptr,
) -> Int:
    var iters = raw_distance(
        a, na, atx, aty, ac, ass, b, nb, btx, bty, bc, bs, result
    )
    apply_radii(ra, rb, use_radii, result)
    return iters


@export("mb2_distance")
def mb2_distance(
    a_addr: Int, na: Int, ra: Float64,
    atx: Float64, aty: Float64, ac: Float64, ass: Float64,
    b_addr: Int, nb: Int, rb: Float64,
    btx: Float64, bty: Float64, bc: Float64, bs: Float64,
    use_radii: Int, result_addr: Int,
) abi("C") -> Int:
    return distance(
        Ptr(unsafe_from_address=a_addr), na, ra, atx, aty, ac, ass,
        Ptr(unsafe_from_address=b_addr), nb, rb, btx, bty, bc, bs,
        use_radii != 0, Ptr(unsafe_from_address=result_addr),
    )


@export("mb2_distance_packed")
def mb2_distance_packed(
    a_addr: Int, b_addr: Int, params_addr: Int,
    use_radii: Int, result_addr: Int,
) abi("C") -> Int:
    var params = Ptr(unsafe_from_address=params_addr)
    return distance(
        Ptr(unsafe_from_address=a_addr), Int(params[0]), params[1],
        params[2], params[3], params[4], params[5],
        Ptr(unsafe_from_address=b_addr), Int(params[6]), params[7],
        params[8], params[9], params[10], params[11],
        use_radii != 0, Ptr(unsafe_from_address=result_addr),
    )


@export("mb2_batch_distance")
def mb2_batch_distance(
    a_addr: Int, na: Int, ra: Float64, ta_addr: Int,
    b_addr: Int, nb: Int, rb: Float64, tb_addr: Int,
    n: Int, use_radii: Int, result_addr: Int,
) abi("C"):
    var a = Ptr(unsafe_from_address=a_addr)
    var b = Ptr(unsafe_from_address=b_addr)
    var ta = Ptr(unsafe_from_address=ta_addr)
    var tb = Ptr(unsafe_from_address=tb_addr)
    var result = Ptr(unsafe_from_address=result_addr)
    for i in range(n):
        var aa = ta[3 * i + 2]
        var ba = tb[3 * i + 2]
        _ = distance(
            a, na, ra, ta[3 * i], ta[3 * i + 1], cos(aa), sin(aa),
            b, nb, rb, tb[3 * i], tb[3 * i + 1], cos(ba), sin(ba),
            use_radii != 0, result + 5 * i,
        )


@export("mb2_test_point")
def mb2_test_point(
    kind: Int, v_addr: Int, n: Int, radius: Float64,
    tx: Float64, ty: Float64, angle: Float64, qx: Float64, qy: Float64,
) abi("C") -> Int:
    var v = Ptr(unsafe_from_address=v_addr)
    var c = cos(angle)
    var s = sin(angle)
    if kind == 0:
        var dx = qx - px(v, 0, tx, c, s)
        var dy = qy - py(v, 0, ty, c, s)
        return 1 if dx * dx + dy * dy <= radius * radius else 0
    return 1 if point_in_polygon(v, n, tx, ty, c, s, qx, qy) else 0


@export("mb2_compute_aabb")
def mb2_compute_aabb(
    v_addr: Int, n: Int, radius: Float64,
    tx: Float64, ty: Float64, angle: Float64, result_addr: Int,
) abi("C"):
    var v = Ptr(unsafe_from_address=v_addr)
    var result = Ptr(unsafe_from_address=result_addr)
    var c = cos(angle)
    var s = sin(angle)
    var lo_x = px(v, 0, tx, c, s)
    var lo_y = py(v, 0, ty, c, s)
    var hi_x = lo_x
    var hi_y = lo_y
    for i in range(1, n):
        var x = px(v, i, tx, c, s)
        var y = py(v, i, ty, c, s)
        lo_x = min(lo_x, x)
        lo_y = min(lo_y, y)
        hi_x = max(hi_x, x)
        hi_y = max(hi_y, y)
    result[0] = lo_x - radius
    result[1] = lo_y - radius
    result[2] = hi_x + radius
    result[3] = hi_y + radius


@export("mb2_batch_aabb_overlap")
def mb2_batch_aabb_overlap(a_addr: Int, b_addr: Int, n: Int, result_addr: Int) abi("C"):
    var a = Ptr(unsafe_from_address=a_addr)
    var b = Ptr(unsafe_from_address=b_addr)
    var result = IPtr(unsafe_from_address=result_addr)
    for i in range(n):
        var k = 4 * i
        result[i] = Int64(
            not (
                b[k] > a[k + 2] or b[k + 2] < a[k]
                or b[k + 1] > a[k + 3] or b[k + 3] < a[k + 1]
            )
        )


@export("mb2_mass")
def mb2_mass(
    kind: Int, v_addr: Int, n: Int, radius: Float64,
    density: Float64, result_addr: Int,
) abi("C"):
    var v = Ptr(unsafe_from_address=v_addr)
    var result = Ptr(unsafe_from_address=result_addr)
    if kind == 0:
        var mass = density * 3.14159265358979323846 * radius * radius
        result[0] = mass
        result[1] = v[0]
        result[2] = v[1]
        result[3] = mass * (0.5 * radius * radius + v[0] * v[0] + v[1] * v[1])
        return
    var twice_area = 0.0
    var cx6 = 0.0
    var cy6 = 0.0
    var inertia12 = 0.0
    for i in range(n):
        var j = (i + 1) % n
        var x1 = v[2 * i]
        var y1 = v[2 * i + 1]
        var x2 = v[2 * j]
        var y2 = v[2 * j + 1]
        var cross = x1 * y2 - y1 * x2
        twice_area += cross
        cx6 += (x1 + x2) * cross
        cy6 += (y1 + y2) * cross
        inertia12 += cross * (
            x1 * x1 + x1 * x2 + x2 * x2
            + y1 * y1 + y1 * y2 + y2 * y2
        )
    var area = 0.5 * twice_area
    result[0] = density * area
    result[1] = cx6 / (3.0 * twice_area)
    result[2] = cy6 / (3.0 * twice_area)
    result[3] = density * inertia12 / 12.0


@export("mb2_ray_cast")
def mb2_ray_cast(
    kind: Int, v_addr: Int, n: Int, radius: Float64,
    tx: Float64, ty: Float64, angle: Float64,
    p1x: Float64, p1y: Float64, p2x: Float64, p2y: Float64,
    max_fraction: Float64, result_addr: Int,
) abi("C") -> Int:
    var v = Ptr(unsafe_from_address=v_addr)
    var result = Ptr(unsafe_from_address=result_addr)
    var c = cos(angle)
    var s = sin(angle)
    if kind == 0:
        var cx = px(v, 0, tx, c, s)
        var cy = py(v, 0, ty, c, s)
        var dx = p2x - p1x
        var dy = p2y - p1y
        var mx = p1x - cx
        var my = p1y - cy
        var a = dx * dx + dy * dy
        var b = mx * dx + my * dy
        var cc = mx * mx + my * my - radius * radius
        var disc = b * b - a * cc
        if disc < 0.0 or a <= 0.0:
            return 0
        var fraction = (-b - sqrt(disc)) / a
        if fraction < 0.0 or fraction > max_fraction:
            return 0
        var hx = p1x + fraction * dx
        var hy = p1y + fraction * dy
        var nl = sqrt((hx - cx) * (hx - cx) + (hy - cy) * (hy - cy))
        result[0] = fraction
        result[1] = (hx - cx) / nl
        result[2] = (hy - cy) / nl
        return 1

    var lower = 0.0
    var upper = max_fraction
    var hit_nx = 0.0
    var hit_ny = 0.0
    var dx = p2x - p1x
    var dy = p2y - p1y
    for i in range(n):
        var j = (i + 1) % n
        var ax = px(v, i, tx, c, s)
        var ay = py(v, i, ty, c, s)
        var bx = px(v, j, tx, c, s)
        var by = py(v, j, ty, c, s)
        var ex = bx - ax
        var ey = by - ay
        var length = sqrt(ex * ex + ey * ey)
        var nx = ey / length
        var ny = -ex / length
        var numerator = nx * (ax - p1x) + ny * (ay - p1y)
        var denominator = nx * dx + ny * dy
        if abs(denominator) < 1.0e-14:
            if numerator < 0.0:
                return 0
            continue
        var fraction = numerator / denominator
        if denominator < 0.0:
            if fraction > lower:
                lower = fraction
                hit_nx = nx
                hit_ny = ny
        elif fraction < upper:
            upper = fraction
        if upper < lower:
            return 0
    if lower < 0.0 or lower > max_fraction:
        return 0
    result[0] = lower
    result[1] = hit_nx
    result[2] = hit_ny
    return 1
