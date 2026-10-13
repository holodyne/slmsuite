"""
Unit tests for the TCP/IP hardware interface: the server in `slmsuite.hardware.remote`
and the `RemoteSLM` / `RemoteCamera` clients, which have no contract apart from it.

Every connection here is over loopback to a server running in this process, so the suite
needs no network and no second machine.
"""

import gc
import json
import logging
import socket
import threading
import time
import warnings
from datetime import date, datetime, timedelta

import numpy as np
import pytest
from conftest import SensorCamera

from slmsuite.hardware.cameras.remote import RemoteCamera
from slmsuite.hardware.cameras.simulated import SimulatedCamera
from slmsuite.hardware.remote import (
    Server,
    _Client,
    _delim,
    _NpEncoder,
    _recurse_decompress,
    _recv,
)
from slmsuite.hardware.slms.remote import RemoteSLM
from slmsuite.hardware.slms.simulated import SimulatedSLM
from slmsuite.holography.toolbox.phase import blaze
from slmsuite.misc.xp import as_numpy

# The server is IPv4-only, and "localhost" can offer it ::1 first.
HOST = "127.0.0.1"


def _free_port():
    """A port the OS has just handed out, on the every-interface address `Server` binds."""
    with socket.socket() as probe:
        probe.bind(("", 0))
        return probe.getsockname()[1]


def _await_listening(server):
    """Blocks until the server has bound its port, so a client cannot race it."""
    deadline = time.perf_counter() + 10
    while not server._listening and time.perf_counter() < deadline:
        time.sleep(0.01)
    assert server._listening, "the server never reached its accept loop"


@pytest.fixture
def hardware():
    """A small simulated SLM and camera, named for a server to host them under."""
    slm = SimulatedSLM((32, 24), pitch_um=(8, 8), name="test_slm")
    cam = SimulatedCamera(slm, (16, 12), pitch_um=(4, 4), name="test_camera")
    yield (slm, cam)
    cam.close()
    slm.close()


@pytest.fixture
def speckle(hardware):
    """The hardware with a speckle phase displayed, so no frame is blank."""
    (slm, cam) = hardware
    slm.set_phase(np.random.default_rng(0).uniform(0, 2 * np.pi, slm.shape))
    cam.gain = 30
    return hardware


@pytest.fixture
def serve():
    """Runs a `Server` in a background thread, returning the port it listens on."""
    running = []

    def start(hardware, **kwargs):
        server = Server(list(hardware), port=_free_port(), timeout=0.05, **kwargs)
        thread = threading.Thread(target=server.listen, daemon=True)
        thread.start()
        running.append((server, thread))
        _await_listening(server)

        return server.port

    yield start

    for server, thread in running:
        server.stop()
        thread.join(timeout=10)
        assert not thread.is_alive(), "the server did not stop within its timeout"


def test_recurse_decompress(subtests):
    """The wire encoding of a message, which compresses arrays and passes the rest through."""
    payload = {
        "image": np.arange(6, dtype=np.uint16).reshape(2, 3),
        "nested": [{"empty": np.zeros((0,))}],
        "scalar": np.int64(7),
        "none": None,
    }
    result = _recurse_decompress(json.loads(json.dumps(payload, cls=_NpEncoder)))

    with subtests.test("arrays survive at any depth, with their dtype and shape"):
        for actual, expected in [
            (result["image"], payload["image"]),
            (result["nested"][0]["empty"], payload["nested"][0]["empty"]),
        ]:
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)

    with subtests.test("non-array data is left alone"):
        assert result["scalar"] == 7
        assert result["none"] is None

    with subtests.test("a request can name a dtype by its numpy scalar type"):
        encoded = json.dumps({"dtype": np.float64}, cls=_NpEncoder)
        assert _recurse_decompress(json.loads(encoded)) == {"dtype": np.dtype(np.float64)}


def test_recv(subtests):
    """A message cut off by a hang-up or by silence is reported, never decoded."""
    for label, hang_up, reason in (
        ("a peer that hangs up mid-message is reported as closed", True, "Connection closed"),
        ("a peer that falls silent mid-message is reported as a timeout", False, "Timeout"),
    ):
        with subtests.test(label):
            (near, far) = socket.socketpair()
            try:
                near.settimeout(7)
                far.sendall(b"partial")
                if hang_up:
                    far.close()
                (success, message) = _recv(near, timeout=0.2)
                assert success is False
                assert message.startswith(reason) and "7 bytes" in message
                assert near.gettimeout() == 7, "the socket's own timeout is restored"
            finally:
                near.close()
                far.close()


class TestNpEncoder:
    def test_default(self, subtests):
        """numpy scalars and dates become JSON."""

        def encode(value):
            return json.loads(json.dumps(value, cls=_NpEncoder))

        with subtests.test("numpy scalars become Python scalars of the same value"):
            encoded = encode([np.bool_(True), np.float32(0.5), np.int8(-3)])
            assert encoded == [True, 0.5, -3]
            assert [type(v) for v in encoded] == [bool, float, int]

        with subtests.test("dates are ISO 8601 and durations are H:MM:SS"):
            assert encode([date(2026, 10, 11), datetime(2026, 10, 11, 12, 30)]) == [
                "2026-10-11",
                "2026-10-11T12:30:00",
            ]
            assert encode(timedelta(seconds=90)) == "0:01:30"


class TestServer:
    def test_init(self, hardware, subtests):
        """What a server accepts to host, and what it refuses."""
        (slm, cam) = hardware

        with subtests.test("hardware is indexed by name and sorted by kind"):
            server = Server(list(hardware), port=_free_port())
            assert server.hardware == {"test_slm": slm, "test_camera": cam}
            assert server.kind == {"test_slm": "slm", "test_camera": "camera"}

        with subtests.test("names must exist and be unique"):
            with pytest.raises(ValueError, match="name"):
                Server([object()], port=_free_port())
            with pytest.raises(ValueError, match="unique"):
                Server([slm, slm], port=_free_port())

        with subtests.test("only cameras and SLMs can be served"):

            class _Neither:
                name = "neither"

            with pytest.raises(ValueError, match="camera or an SLM"):
                Server([_Neither()], port=_free_port())

        with subtests.test("the port must be an unprivileged one in range"):
            for port in (80, 70000):
                with pytest.raises(ValueError, match=r"[Pp]ort"):
                    Server([slm], port=port)

    def test_identify_hardware(self, hardware):
        """Hardware is sorted by the abstract method it implements, not by its class."""
        (slm, cam) = hardware
        assert Server.identify_hardware(cam) == "camera"
        assert Server.identify_hardware(slm) == "slm"
        assert Server.identify_hardware(object()) is None

    def test_handle(self, hardware, subtests):
        """Message dispatch, which is where an untrusted peer is held at arm's length."""
        (_, cam) = hardware
        server = Server(list(hardware), port=_free_port())

        with subtests.test("'ping' answers with the hardware on offer"):
            assert server._handle({"command": "ping"}) == (True, server.kind)

        with subtests.test("a message without a command is refused"):
            assert server._handle({})[0] is False

        with subtests.test("unknown hardware is named back with the options"):
            (success, reply) = server._handle({"name": "absent", "command": "pickle"})
            assert success is False
            assert "test_slm" in reply

        with subtests.test("a command outside the allowlist is refused"):
            for command in ("close", "set_phase", "__init__"):
                assert server._handle({"name": "test_slm", "command": command})[0] is False

        with subtests.test("an allowed command runs on the named hardware"):
            (success, reply) = server._handle(
                {"name": "test_camera", "command": "_get_image_hw", "kwargs": {"timeout_s": 1}}
            )
            assert success is True
            assert np.shape(reply) == cam.shape

        with subtests.test("a client cannot choose which attributes pickle reads"):
            (_, chosen) = server._handle(
                {
                    "name": "test_slm",
                    "command": "pickle",
                    "kwargs": {"attributes": ["name"]},
                }
            )
            (_, everything) = server._handle(
                {
                    "name": "test_slm",
                    "command": "pickle",
                    "kwargs": {"attributes": True},
                }
            )
            assert set(chosen["__meta__"]) == set(everything["__meta__"])

        with subtests.test("a rejected keyword names the argument"):
            (success, reply) = server._handle(
                {"name": "test_camera", "command": "_get_image_hw", "kwargs": {"nonsense": 1}}
            )
            assert success is False
            assert "nonsense" in reply

        with subtests.test("an error inside the hardware does not reach the peer"):
            cam._get_exposure_hw = lambda: 1 / 0

            (success, reply) = server._handle(
                {"name": "test_camera", "command": "_get_exposure_hw"}
            )
            assert success is False
            assert "Traceback" not in reply and "ZeroDivision" not in reply

    def test_listen(self, hardware, serve, caplog, subtests):
        """The socket loop, which serves one request per connection and survives a bad one."""
        port = serve(hardware)

        with subtests.test("a client reaches the hosted hardware"):
            assert _Client.info(host=HOST, port=port, verbose=False) == {
                "test_slm": "slm",
                "test_camera": "camera",
            }

        with subtests.test("a malformed request does not take the server down"):
            with socket.create_connection((HOST, port), timeout=10) as sock:
                sock.settimeout(10)
                sock.sendall(b"not json at all\n")
                sock.recv(4096)

            assert _Client.info(host=HOST, port=port, verbose=False)["test_slm"] == "slm"

        with subtests.test("a request cut off mid-message is answered with the reason"):
            with socket.create_connection((HOST, port), timeout=10) as sock:
                sock.sendall(b"partial")
                sock.shutdown(socket.SHUT_WR)
                reply = _recv(sock, timeout=10)
            assert reply == [False, "Connection closed by peer after 7 bytes received."]

        with subtests.test("a client outside the allowlist is turned away"):
            blocked = serve(hardware, allowlist=["10.0.0.1"])

            # The server closes on a rejected peer, which can reset the refusal in transit.
            with caplog.at_level(logging.WARNING, logger="slmsuite.hardware.remote"):
                with pytest.raises((RuntimeError, OSError)):
                    _Client.info(host=HOST, port=blocked, verbose=False)

            assert [
                record
                for record in caplog.records
                if record.levelno == logging.WARNING and "allowlist" in record.message
            ], "the server should record the rejection it made"

    def test_stop(self, hardware):
        """`stop()` ends the listen loop and closes the socket behind it."""
        server = Server(list(hardware), port=_free_port(), timeout=0.05)
        thread = threading.Thread(target=server.listen, daemon=True)
        thread.start()
        _await_listening(server)

        assert _Client.info(host=HOST, port=server.port, verbose=False)

        server.stop()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert not server._listening

        with pytest.raises(TimeoutError, match="not responsive"):
            _Client.info(host=HOST, port=server.port, timeout=0.2, verbose=False)


def test_info(hardware, serve, subtests):
    """What `info()` reports about a port, which is how a client finds its hardware."""
    port = serve(hardware)

    with subtests.test("a served port lists its hardware by kind"):
        assert _Client.info(host=HOST, port=port, verbose=False) == {
            "test_slm": "slm",
            "test_camera": "camera",
        }

    with subtests.test("an unserved port is reported as absent"):
        with pytest.raises(TimeoutError, match="not responsive"):
            _Client.info(host=HOST, port=_free_port(), timeout=0.2, verbose=False)

    with subtests.test("a failed connect closes its socket"):
        gc.collect()  # Else garbage from earlier tests can warn inside the block.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            with pytest.raises(TimeoutError, match="not responsive"):
                _Client.info(host=HOST, port=_free_port(), timeout=0.2, verbose=False)
            gc.collect()
        assert not [w for w in caught if issubclass(w.category, ResourceWarning)]

    with subtests.test("a listener that is not a server is not reported as absent"):
        # DEFAULT_PORT is the usual instrument-control port, so something else may answer.
        other = socket.socket()
        other.bind((HOST, 0))
        other.listen(1)

        def reply_garbage():
            (connection, _) = other.accept()
            try:
                connection.recv(65536)
                connection.sendall(b"not an slmsuite reply" + _delim.encode())
            finally:
                connection.close()

        thread = threading.Thread(target=reply_garbage, daemon=True)
        thread.start()
        try:
            with pytest.raises(Exception) as excinfo:  # noqa: PT011 (any failure but a timeout)
                _Client.info(host=HOST, port=other.getsockname()[1], verbose=False)
            assert not isinstance(excinfo.value, TimeoutError)
        finally:
            thread.join(timeout=10)
            other.close()


class TestRemoteSLM:
    def test_init(self, hardware, serve, subtests):
        """The client reads its geometry from the server it connects to."""
        (slm, _) = hardware
        port = serve(hardware)

        with subtests.test("the client mirrors the server's geometry"):
            remote = RemoteSLM(name="test_slm", host=HOST, port=port)
            assert remote.shape == slm.shape
            assert remote.bitdepth == slm.bitdepth
            np.testing.assert_array_equal(remote.pitch_um, slm.pitch_um)
            assert remote.wav_um == slm.wav_um

        with subtests.test("wav_um and settle_time_s override what the server reports"):
            remote = RemoteSLM(
                name="test_slm",
                host=HOST,
                port=port,
                wav_um=2 * slm.wav_um,
                settle_time_s=0.25,
            )
            assert remote.wav_um == 2 * slm.wav_um
            assert remote.settle_time_s == 0.25

        with subtests.test("a name that is absent or not an SLM is refused"):
            with pytest.raises(ValueError, match="not present"):
                RemoteSLM(name="absent", host=HOST, port=port)
            with pytest.raises(ValueError, match="not a slm"):
                RemoteSLM(name="test_camera", host=HOST, port=port)

    def test_set_phase_hw(self, hardware, serve):
        """The forwarder's whole contract: the display arrives on the server unaltered."""
        (slm, _) = hardware
        received = []
        slm._set_phase_hw = lambda display, **kwargs: received.append(display)

        remote = RemoteSLM(name="test_slm", host=HOST, port=serve(hardware))
        remote.set_phase(blaze(remote, vector=(0.01, -0.005)), settle=False)

        assert len(received) == 1
        assert received[0].dtype == remote.display.dtype
        np.testing.assert_array_equal(as_numpy(received[0]), as_numpy(remote.display))


class TestRemoteCamera:
    def test_init(self, hardware, serve, caplog, subtests):
        """The client reads its sensor from the server it connects to."""
        (_, cam) = hardware
        port = serve(hardware)
        pickle = cam.pickle

        with subtests.test("the client mirrors the server's sensor"):
            remote = RemoteCamera(name="test_camera", host=HOST, port=port)
            assert remote.shape == cam.shape
            assert remote.bitdepth == cam.bitdepth
            np.testing.assert_array_equal(remote.pitch_um, cam.pitch_um)

        with subtests.test("a name that is not a camera is refused"):
            with pytest.raises(ValueError, match="not a camera"):
                RemoteCamera(name="test_slm", host=HOST, port=port)

        for label, served, message in (
            ("another version", lambda **kw: {**pickle(**kw), "__version__": "0.0"}, "0.0"),
            (
                "no version",
                lambda **kw: {"__meta__": pickle(attributes=False, metadata=False)},
                "did not provide version",
            ),
        ):
            with subtests.test(f"a server reporting {label} is connected to, with a warning"):
                cam.pickle = served
                caplog.clear()
                with caplog.at_level(logging.WARNING, logger="slmsuite.hardware.remote"):
                    remote = RemoteCamera(name="test_camera", host=HOST, port=port)
                assert remote.shape == cam.shape
                assert message in caplog.text

        with subtests.test("a server that cannot describe its camera is not connected to"):
            cam.pickle = lambda **kw: 1 / 0
            with pytest.raises(RuntimeError, match="Could not connect"):
                RemoteCamera(name="test_camera", host=HOST, port=port)

    def test_get_image_hw(self, hardware, serve):
        """The image the server renders is the image the client receives, dtype included."""
        (_, cam) = hardware
        remote = RemoteCamera(name="test_camera", host=HOST, port=serve(hardware))

        expected = cam.get_image()
        image = remote.get_image()
        assert image.dtype == expected.dtype
        np.testing.assert_array_equal(image, expected)

    def test_set_exposure_hw(self, hardware, serve):
        """Exposure lives on the server, so the client reads back what it wrote."""
        (_, cam) = hardware
        remote = RemoteCamera(name="test_camera", host=HOST, port=serve(hardware))

        remote.set_exposure(0.25)
        assert remote.get_exposure() == 0.25
        assert cam.get_exposure() == 0.25

    def test_get_images_hw(self, speckle, serve, caplog, subtests):
        """A stack fetched from the server is the stack the server captures."""
        (_, cam) = speckle
        remote = RemoteCamera(name="test_camera", host=HOST, port=serve(speckle))
        expected = cam.get_images(3)
        assert np.ptp(expected) > 0

        with subtests.test("the stack arrives with its dtype and shape"):
            stack = remote.get_images(3)
            assert stack.dtype == expected.dtype
            np.testing.assert_array_equal(stack, expected)

        with subtests.test("averaging sums a stack fetched in one request"):
            np.testing.assert_array_equal(remote.get_image(averaging=3), np.sum(expected, axis=0))

        with subtests.test("a buffer is filled, though the transfer cannot be in place"):
            out = np.zeros_like(expected)
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="slmsuite"):
                assert remote.get_images(3, out=out) is out
            np.testing.assert_array_equal(out, expected)
            assert "in-place" in caplog.text

    def test_get_woi_hw(self, speckle, serve, subtests):
        """A server that windows and bins in software leaves both to the client."""
        (_, cam) = speckle
        remote = RemoteCamera(name="test_camera", host=HOST, port=serve(speckle))
        assert remote._software_woi and remote._software_binning

        for c in (remote, cam):
            c.set_woi((2, 12, 2, 8))
            c.set_binning(2)

        with subtests.test("the client crops and bins the full frame as the server would"):
            assert remote.shape == cam.shape == (4, 6)
            np.testing.assert_array_equal(remote.get_image(), cam.get_image())

        with subtests.test("the window and binning read back from the client"):
            assert remote.get_woi() == (2, 12, 2, 8) and remote.get_binning() == (2, 2)

    def test_set_woi_hw(self, speckle, serve, subtests):
        """A server that windows in hardware is handed the window, and returns the crop."""
        (slm, cam) = speckle
        windowed = SensorCamera(slm, (16, 12), pitch_um=(4, 4), gain=cam.gain, name="win")
        windowed.snap = 2
        remote = RemoteCamera(name="win", host=HOST, port=serve([slm, cam, windowed]))
        assert not remote._software_woi

        remote.set_woi((5, 8, 3, 6))
        cam.set_woi((4, 8, 2, 6))

        with subtests.test("the window is set on the server's hardware and read back from it"):
            assert windowed._get_woi_hw() == (4, 8, 2, 6)
            assert remote.woi == (4, 8, 2, 6) == remote.get_woi()

        with subtests.test("the frame is the hardware crop, the software crop's equal"):
            assert remote.shape == (6, 8)
            np.testing.assert_array_equal(remote.get_image(), cam.get_image())
