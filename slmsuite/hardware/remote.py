"""
Provides a TCPIP client-server interface to control remote hardware.
Interface with a :class:`~slmsuite.hardware.remote.Server` using
:class:`~slmsuite.hardware.slms.remote.RemoteSLM` and
:class:`~slmsuite.hardware.cameras.remote.RemoteCamera`.
Under the hood, this uses :mod:`json` formatting with
:mod:`zlib` compression used for any :mod:`numpy` data, including especially
megapixel phase masks and megapixel camera images.

Danger
~~~~~~
This interface will operate best in a secure and trusted local network.
Hosting a server on a public IP raises some risk of tampering
(even with an ``allowlist``, as IP addresses can be spoofed).
Communication is locked down to only the few commands that are required
by an abstract camera or SLM to reduce risk.
However, there are no protections against DDOS attack or similar and
communication is **not encrypted or authenticated**.

In the future, if there is interest, SSH authentication could be added, and the scope of
commands offered could be expanded beyond the required abstract commands
to also include whatever hardware-specific functionality is
implemented. Moreover, CameraSLMs could be hosted on a server such that system
calibrations could live in the cloud.

Example
~~~~~~~
The following is a simple example of a setup communicating between two
threads---for example, two Jupyter notebooks---on the same computer (via ``localhost:5025``).

The server hosts a simulated SLM and camera:

.. highlight:: python
.. code-block:: python

    # Server notebook
    from slmsuite.hardware.slms.simulated import SimulatedSLM
    from slmsuite.hardware.cameras.simulated import SimulatedCamera
    from slmsuite.hardware.remote import Server

    slm = SimulatedSLM((1600, 1200), pitch_um=(8,8), name="remote_slm")
    cam = SimulatedCamera(slm, (1440, 1100), pitch_um=(4,4), gain=50, name="remote_camera")

    server = Server(
        hardware=[slm, cam],
        port=5025,
    )

The client connects to this hardware:

.. highlight:: python
.. code-block:: python

    # Client notebook
    from slmsuite.hardware.slms.remote import RemoteSLM
    from slmsuite.hardware.cameras.remote import RemoteCamera

    slm = RemoteSLM(
        name="remote_slm",
        host="localhost",
        port=5025,
    )
    cam = RemoteCamera(
        name="remote_camera",
        host="localhost",
        port=5025,
    )

    slm.set_phase(None)

    cam.get_image()
    cam.plot()
"""
import base64
import json
import socket
import time
import traceback
import urllib.parse as urllib
import zlib
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np

from slmsuite import __version__
from slmsuite._logging import make_logger
from slmsuite._pickling import _Picklable

logger = make_logger(__name__)

DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 5025             # Commonly used for instrument control.
DEFAULT_TIMEOUT = 5
SERVER_WAIT_TIMEOUT = 0.5

_delim = "\n"

# Common functions for encoding and decoding data.
def _recurse_decompress(msg):
    """
    Recursively decompresses the result of json serialization.
    """
    if isinstance(msg, dict):
        if "__zlib__" in msg and len(msg) == 3:
            return np.frombuffer(
                zlib.decompress(
                    base64.b64decode(msg["__zlib__"])
                ),
                dtype=np.dtype(msg["__dtype__"])
            ).reshape(msg["__shape__"])
        elif "__dtype__" in msg and len(msg) == 1:
            return np.dtype(msg["__dtype__"])
        else:
            for k in msg:
                msg[k] = _recurse_decompress(msg[k])
    elif isinstance(msg, list):
        for i, m in enumerate(msg):
            msg[i] = _recurse_decompress(m)

    return msg

# https://codetinkering.com/numpy-encoder-json/
class _NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.floating): #, np.complexfloating
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return {
                "__zlib__" : base64.b64encode(
                    zlib.compress(obj.tobytes())
                ).decode(),
                "__shape__" : obj.shape,
                "__dtype__" : str(obj.dtype)
            }
        if isinstance(obj, np.string_):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, timedelta):
            return str(obj)
        if isinstance(obj, np.dtype):
            return {"__dtype__" : str(obj)}
        return super(_NpEncoder, self).default(obj)

# Common function to receive data from a socket.
def _recv(sock, timeout):
    recv_buffer = 4096 * 64
    buffer = ""
    t = time.time()

    # Pull data into the buffer until we hit timeout or deliminator.
    while time.time() - t < timeout:
        data = sock.recv(recv_buffer).decode()

        buffer += data
        if data[-1] == _delim:
            msg = json.loads(urllib.unquote_plus(buffer[0:-len(_delim)]))

            msg = _recurse_decompress(msg)

            return msg

    # Failed timeout returns empty.
    return False, f"Timeout: {len(buffer)} bytes received."

# Server which hosts slmsuite hardware.
class Server(object):
    """
    Server for handling client commands and interfacing with hardware.
    """

    def __init__(
            self,
            hardware: List[object],
            port: int = DEFAULT_PORT,
            timeout: float = SERVER_WAIT_TIMEOUT,
            allowlist: List[str] = None,
        ):
        """
        Initializes a server to host slmsuite hardware: cameras and SLMs.
        Interface with this server using
        :class:`~slmsuite.hardware.slms.remote.RemoteSLM` and
        :class:`~slmsuite.hardware.cameras.remote.RemoteCamera`.

        :param hardware:
            List of hardware objects to serve.
        :param port:
            Port number to serve on. Defaults to ``5025``
            (commonly used for instrument control).
        :param timeout:
            Timeout in seconds for the server to wait for a client.
        :param allowlist:
            List of IP addresses to allow to connect. Defaults to ``None`` (allow all).
            Keep in mind that IP addresses can be spoofed, so this ``allowlist``
            provides only modest security.
        """
        # Parse hardware.
        for hw in hardware:
            if not hasattr(hw, "name"):
                raise ValueError(f"Hardware {hw} must have a 'name' attribute.")
            if self.identify_hardware(hw) is None:
                raise ValueError(f"Hardware {hw.name} ({hw}) must be either a camera or an SLM.")

        names = [hw.name for hw in hardware]
        if len(set(names)) != len(names):
            raise ValueError(f"Hardware names must be unique. Found {names}.")

        self.hardware = {
            hw.name : hw
            for hw in hardware
        }
        self.kind = {
            hw.name : self.identify_hardware(hw)
            for hw in hardware
        }

        # Server information.
        if not (1024 <= port <= 65535):
            raise ValueError(f"Invalid port number: {port}. Use a port between 1024 and 65535.")
        self.port = port
        self.timeout = timeout

        # Only allow clients in the allowlist to connect.
        self.allowlist = allowlist

        # Only allowed commands for SLMs and Cameras, alongside "ping".
        self.allowcommands = [
            "pickle",
            "flush",
            "_set_phase_hw",
            "_set_exposure_hw",
            "_get_exposure_hw",
            "_get_image_hw",
            "_get_images_hw",
            "_set_woi_hw",
            "_get_woi_hw",
            "_set_binning_hw",
            "_get_binning_hw",
        ]

    @staticmethod
    def identify_hardware(hw: object) -> str:
        if hasattr(hw, "_get_image_hw"):
            return "camera"
        elif hasattr(hw, "_set_phase_hw"):
            return "slm"
        else:
            return None

    def listen(self):
        """
        Blocking command to listen for client commands and process them once they are
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(self.timeout)
        sock.bind(("", self.port))
        sock.listen(5)

        logger.info("Hosting on port %s with hardware %s", self.port, list(self.hardware.keys()))

        try:
            while True:
                try:
                    # This blocks for self.timeout unless a client connects.
                    connection, client_addr = sock.accept()

                    # Check if the client is allowed to connect.
                    if (self.allowlist is not None) and (client_addr[0] not in self.allowlist):
                        logger.warning(
                            "Rejected connection from %s, not in allowlist %s.",
                            client_addr, self.allowlist,
                        )
                        result = False, f"Client {client_addr} not in allowlist."
                    else:
                        # Receive, handle, and reply to message.
                        message = _recv(connection, self.timeout)
                        result = self._handle(message, client_addr)

                    reply = (urllib.quote_plus(json.dumps(result, cls=_NpEncoder)) + _delim).encode()
                    logger.debug("Replied with %d bytes.", len(reply))

                    connection.sendall(reply)
                    connection.close()
                except IOError:
                    # This is a timeout error. Just continue.
                    pass
                except Exception as e:
                    # Pass to the outer try for all other errors.
                    raise e
        except KeyboardInterrupt:
            # Standard way to kill the thread.
            logger.info("Closing server! Goodbye!")
            try:
                connection.close()
            except:
                pass
            sock.close()
        except Exception as e:
            # There was an error in the server communication protocol. This kills the thread.
            # Note that hardware errors are handled in _handle and the loop continues.
            logger.error("Server communication error:\n%s", traceback.format_exc())
            try:
                connection.close()
            except:
                pass
            sock.close()
            raise e

    def _handle(
        self,
        message : str,
        client_addr: str = None,
    ) -> Tuple[bool, Any]:
        """
        Handle a message from a client.
        """
        try:
            name = message.pop("name", None)
            command = message.pop("command", None)
            args = message.pop("args", [])
            kwargs = message.pop("kwargs", dict())

            instrument = f"{name}.{command}"

            logger.info("%s %s", client_addr, instrument)

            # Initial parse of command.
            if command is None:
                return False, "No command provided."
            elif command == "ping":
                return True, self.kind

            # Make sure that the hardware exists.
            if name not in self.hardware:
                return False, f"Did not recognize hardware '{name}'. Options: {list(self.hardware.keys())}."

            if command in self.allowcommands and hasattr(self.hardware[name], command):
                attribute = getattr(self.hardware[name], command)
                if callable(attribute):
                    return True, attribute(*args, **kwargs)
                else:
                    return False, f"{instrument} is not callable."
            else:
                return False, f"{instrument} not present."
        except:
            return False, traceback.format_exc()

# Abstract client which connects to a server.
class _Client(_Picklable):
    """
    Client for interfacing with a slmsuite server.
    """

    def __init__(
            self,
            name: str,
            kind: str,
            host: str = DEFAULT_HOST,
            port: int = DEFAULT_PORT,
            timeout: float = DEFAULT_TIMEOUT,
        ):
        """
        Superclass constructor. See RemoteSLM and RemoteCamera for more information.
        """
        self.name = name
        self.host = host
        self.port = port
        self.timeout = timeout

        hardware = self._com(command="ping")

        if self.name not in hardware:
            raise ValueError(
                f"Hardware '{self.name}' is not present at {self.host}:{self.port}. Options: {hardware}."
            )
        if hardware[self.name] != kind:
            raise ValueError(
                f"Hardware '{self.name}' is not a {kind} at {self.host}:{self.port}."
            )

        try:
            t = time.perf_counter()
            pickled = self._com(
                command="pickle",
                kwargs=dict(attributes=False, metadata=True)
            )
            t = time.perf_counter() - t
        except:
            raise RuntimeError(
                f"Could not connect to '{self.name}' at {self.host}:{self.port}. Options: {hardware}."
            )

        self.latency_s = t
        self.server_attributes = pickled

        if "__version__" not in pickled:
            logger.warning(
                "Server did not provide version information; "
                "cannot verify compatibility with client version %s.", __version__
            )
        elif pickled["__version__"] != __version__:
            logger.warning(
                "Client version %s does not match server version %s.",
                __version__, pickled["__version__"],
            )

    def _com(
        self,
        command: str = "ping",
        args: list = [],
        kwargs: dict = {},
    ):
        """Helper function to _com without having to put all the name/host information in."""
        return _Client.__com(self.name, self.host, self.port, self.timeout, command, args, kwargs)

    @staticmethod
    def __com(
        name: str,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = DEFAULT_TIMEOUT,
        command: str = "ping",
        args: list = [],
        kwargs: dict = {},
    ):
        """Generalized function to communicate with a server."""
        # Create a TCP/IP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect((host, port))
        except (TimeoutError, ConnectionRefusedError):
            raise ValueError(
                f"An slmsuite server is not active at {host}:{port}."
            )
        except Exception as e:
            raise e


        # Send the message.
        sock.sendall((
            urllib.quote_plus(
                json.dumps(
                    {
                        "name": name,
                        "command": command,
                        "args": args,
                        "kwargs": kwargs
                    },
                    cls=_NpEncoder
                )
            ) + _delim
        ).encode())


        # Wait for a reply.
        try:
            success, reply = _recv(sock, timeout)
            if success == False:
                raise RuntimeError(
                    f"Server {host}:{port} communication failed. Message:\n{reply}"
                )
        except Exception as e:
            sock.close()

            raise e

        # Always close.
        sock.close()

        return reply

    @staticmethod
    def info(
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = DEFAULT_TIMEOUT,
        verbose: bool = True,
    ) -> Dict[str, str]:
        """
        Looks to see if a slmsuite server is active at the given host and port.

        :param verbose:
            Whether to print the discovered information.
        :param host:
            Which host to connect to. Defaults to ``"localhost"``.
        :param port:
            Which port to connect to. Defaults to ``5025`` (commonly used for instrument
            control).
        :param timeout:
            Timeout in seconds for the connection. Defaults to ``5``.
        :return:
            List of hardware at the server in ``name:kind`` pairs, where ``kind`` is
            either ``"camera"`` or ``"slm"``. Raises :exc:`TimeoutError` if no server is found.
        """
        try:
            hardware = _Client.__com(None, host, port, timeout, command="ping")
        except (TimeoutError, ConnectionRefusedError):
            raise TimeoutError(f"Did not find a server at {host}:{port}.")
        except Exception as e:
            raise e

        if verbose:
            if len(hardware) == 0:
                print(f"Server found at {host}:{port} with no hardware.")
            else:
                print(
                    f"Server found at {host}:{port} with hardware:\n    " +
                    "\n    ".join(list(hardware.keys()))
                )

        return hardware