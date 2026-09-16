"""Receipt-only interval alignment; every conclusion declares its drift assumption."""


def host_shift(markers, brackets):
    items = {x['name']: x for x in brackets}
    lower = min(items[name]['before'] - stamp for name, stamp in markers.items())
    upper = max(items[name]['after'] - stamp for name, stamp in markers.items())
    assert 0 <= upper - lower < 1_000_000, ('trace_host_bridge_over_1ms', upper - lower)
    return lower, upper


def offset_bounds(peer_calibrations, local, ppm):
    candidates = []
    for calibration in peer_calibrations:
        b = calibration['bounds']
        samples = calibration['samples']
        elapsed = max(abs(local - t) for sample in samples for t in sample[1:3])
        error = elapsed * ppm / 1e6
        candidates.append((b['lower_ns'] - error, b['upper_ns'] + error))
    low = max(x[0] for x in candidates)
    high = min(x[1] for x in candidates)
    if low > high:
        return None
    return low, high
