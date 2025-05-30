"""
Runs MLflow server, gateway, and UI in development mode.
"""

# pylint: disable=broad-exception-caught

import os
import sys
import time
import socket
import subprocess
from contextlib import closing


def random_port(bind_address: str = "127.0.0.1", safe: bool = False) -> int:
    """
    Get an available random port bound to a specified network interface.

    This function creates a temporary socket, binds it to the given IP address
    with port `0` to let the OS assign an available ephemeral port, and then
    returns the assigned port number.

    Parameters
    ----------
    bind_address : str, optional
        The IP address to bind the socket to. If `safe` is True and this value
        is empty ('') or '0.0.0.0', it is internally replaced with '127.0.0.1'.
        Defaults to '127.0.0.1'.

    Returns
    -------
    int
        A randomly selected available port number on the local machine.

    Notes
    -----
    Binding to '127.0.0.1' ensures the port is only accessible from the local
    machine, mitigating security risks associated with binding to all
    interfaces (e.g., '' or '0.0.0.0').

    This method is commonly used in testing or ephemeral service spawning
    scenarios where a free local port is needed without exposing it to external
    networks.

    Security:
    - Avoid binding to '' or '0.0.0.0', which exposes the port to all network interfaces.
    - Binding to '127.0.0.1' limits exposure to local access only.

    Examples
    --------
    >>> random_port()
    50321

    >>> random_port('192.168.1.10')
    50405

    >>> random_port('', safe=True)
    50392  # Internally binds to '127.0.0.1'

    >>> random_port('0.0.0.0', safe=True)
    50400  # Also internally binds to '127.0.0.1'

    See Also
    --------
    socket.getsockname : Returns the address and port of the bound socket.
    socket.bind : Binds the socket to an address and port.
    """
    # Binds to all interfaces (0.0.0.0) — insecure
    # Binds only to localhost — secure
    unsafe_addresses = {"", "0.0.0.0"}
    safe_bind_address = (
        "127.0.0.1" if safe and bind_address in unsafe_addresses else bind_address
    )
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((safe_bind_address, 0))
        return s.getsockname()[1]


def is_port_open(host: str, port: int) -> bool:
    """Check if a TCP port is open on the given host."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def wait_for_port(host: str, port: int, timeout: float = 5.0) -> bool:
    """Wait until a port is open, with timeout."""
    start = time.time()
    while time.time() - start < timeout:
        if is_port_open(host, port):
            return True
        time.sleep(0.1)
    return False


def safe_random_port(host: str = "127.0.0.1", retries: int = 10) -> int:
    """Retry getting a free port that is truly available."""
    for _ in range(retries):
        port = random_port(bind_address=host, safe=True)
        if not is_port_open(host, port):  # not already in use
            return port
    raise RuntimeError("Failed to find an available port after multiple retries.")


def main():
    """
    Start the scikitplot gateway, server, and UI subprocesses using a dynamically
    allocated local port, ensuring they are terminated gracefully on interrupt.

    This function:
    - Acquires a random safe port using `random_port()`
    - Launches the gateway on the selected port
    - Starts the backend server with the gateway endpoint passed via environment
    - Runs the frontend UI from the specified JavaScript directory
    - Keeps all subprocesses alive until interrupted via Ctrl+C

    Notes
    -----
    - The gateway is started with a predefined YAML config file.
    - The UI assumes a Yarn-based JavaScript frontend in `scikitplot/server/js`.
    - The `random_port()` function ensures the gateway port is locally bound and avoids conflicts.

    See Also
    --------
    random_port : Gets an available random local port with optional safety controls.

    Examples
    --------
    >>> main()  # Starts all services and waits until manually interrupted
    """
    gateway_host = "localhost"
    gateway_port = safe_random_port(gateway_host)
    try:
        with subprocess.Popen(
            [
                sys.executable,
                "-m",
                "scikitplot",
                "gateway",
                "start",
                "--config-path",
                "examples/gateway/openai/config.yaml",
                "--host",
                gateway_host,
                "--port",
                str(gateway_port),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as gateway:
            if not wait_for_port(gateway_host, gateway_port, timeout=5):
                gateway.terminate()
                raise RuntimeError(
                    f"Gateway failed to start on {gateway_host}:{gateway_port}"
                )

            with (
                subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "scikitplot",
                        "server",
                        "--dev",
                    ],
                    env={
                        **os.environ,
                        "SKPLT_DEPLOYMENTS_TARGET": (
                            f"http://{gateway_host}:{gateway_port}"
                        ),
                    },
                ) as server,
                subprocess.Popen(
                    [
                        "yarn",
                        "start",
                    ],
                    cwd="scikitplot/server/js",
                ) as ui,
            ):
                print(f"Gateway running at http://{gateway_host}:{gateway_port}")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nShutting down services...")
                    server.terminate()
                    ui.terminate()
                    gateway.terminate()
    except Exception as e:
        print(f"Startup error: {e}")


if __name__:
    main()
