"""CPU socket calibration outside training/capture; asymmetric latency intervals."""

import json
import os
import socket
import time


def interval(samples):
    lower = max(t3 - t4 for t1, t2, t3, t4 in samples)
    upper = min(t2 - t1 for t1, t2, t3, t4 in samples)
    assert lower <= upper, ('inconsistent_clock_intervals', lower, upper)
    return dict(lower_ns=lower, upper_ns=upper, width_ns=upper - lower)


def send(stream, value):
    stream.write(json.dumps(value).encode() + b'\n')
    stream.flush()


def receive(stream):
    line = stream.readline(65536)
    assert line.endswith(b'\n')
    return json.loads(line)


def calibrate(phase):
    assert phase in ('pre', 'post')
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    assert world == 16
    host = os.environ['MASTER_ADDR']
    port = int(os.environ['CAUSAL_CLOCK_PORT']) + (phase == 'post')
    result = dict(
        phase=phase,
        rank=rank,
        host=socket.gethostname(),
        monotonic_start=time.monotonic_ns(),
        peers=[],
    )
    deadline = time.monotonic() + 120

    def remaining():
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError(f'clock calibration deadline phase={phase} rank={rank}')
        return value

    if rank == 0:
        with socket.socket() as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(('', port))
            server.listen(16)
            server.settimeout(120)
            seen = set()
            for _ in range(world - 1):
                server.settimeout(remaining())
                connection, _ = server.accept()
                with connection:
                    connection.settimeout(remaining())
                    with connection.makefile('rwb') as stream:
                        hello = receive(stream)
                        peer = hello['rank']
                        assert hello['phase'] == phase and 1 <= peer < world and peer not in seen
                        seen.add(peer)
                        samples = []
                        for i in range(32):
                            connection.settimeout(remaining())
                            t1 = time.monotonic_ns()
                            send(stream, dict(i=i, t1=t1))
                            response = receive(stream)
                            t4 = time.monotonic_ns()
                            assert response['i'] == i
                            samples.append([t1, response['t2'], response['t3'], t4])
                        bounds = interval(samples)
                        send(stream, dict(done=True))
                        result['peers'].append(
                            dict(rank=peer, host=hello['host'], samples=samples, bounds=bounds)
                        )
            assert seen == set(range(1, world))
    else:
        while True:
            try:
                connection = socket.create_connection((host, port), timeout=5)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        with connection:
            connection.settimeout(120)
            with connection.makefile('rwb') as stream:
                send(stream, dict(rank=rank, phase=phase, host=socket.gethostname()))
                for i in range(32):
                    connection.settimeout(remaining())
                    request = receive(stream)
                    t2 = time.monotonic_ns()
                    assert request['i'] == i
                    t3 = time.monotonic_ns()
                    send(stream, dict(i=i, t2=t2, t3=t3))
                assert receive(stream) == dict(done=True)
    result['monotonic_end'] = time.monotonic_ns()
    return result
