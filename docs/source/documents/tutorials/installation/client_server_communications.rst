Client-Server Communications
=============================

.. rubric:: Overview

MOSAIC implements a **distributed client-server architecture** in which the
graphical user interface (client) and the training compute backend (server)
are decoupled processes that communicate over gRPC. This design allows the
GUI to run on any machine with a display while all GPU-intensive workloads
execute on a dedicated compute node — including headless servers, cloud
instances, and multi-GPU lab machines.

Three launch scripts cover every deployment scenario:

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Script
     - Run on
     - Purpose
   * - ``run.sh``
     - Any single machine
     - Starts server + client together (development / single-node)
   * - ``run_server.sh``
     - Compute / GPU node
     - Headless server only (trainer daemon + optional vLLM)
   * - ``run_client.sh``
     - Display machine
     - GUI only — connects to whichever server ``MOSAIC_DAEMON_TARGET`` points at

.. contents:: On this page
   :local:
   :depth: 2

Architecture
------------

MOSAIC's runtime consists of three distinct layers:

.. mermaid::

   graph TB
       classDef client  fill:#1e3d59,stroke:#3498db,color:#aed6f1
       classDef network fill:#1a1a2e,stroke:#5b5bcc,color:#b0b0ff
       classDef server  fill:#0d2b1a,stroke:#27ae60,color:#a8f0c6
       classDef svc     fill:#2d1040,stroke:#8e44ad,color:#d7bde2
       classDef worker  fill:#1a2d3d,stroke:#2980b9,color:#aed6f1

       subgraph CLIENT["🖥  CLIENT  ·  hamid@HamidOnUbuntu  ·  run_client.sh"]
           direction LR
           APP["gym_gui.app / PyQt6"]:::client
           TC["TrainerClient / grpc.aio stub"]:::client
           APP --> TC
       end

       subgraph NETWORK["LAN · gRPC over HTTP/2 · port 50055"]
           WIRE[" "]:::network
       end

       subgraph SERVER["⚡  SERVER  ·  Hamid@a1-R8428-G11  ·  run_server.sh"]
           direction TB
           DM["gym_gui.services.trainer_daemon"]:::server

           subgraph TS["TrainerService  —  gRPC servicer"]
               direction LR
               RPC1["SubmitRun · CancelRun · ListRuns · GetHealth"]:::svc
               RPC2["WatchRuns  server-streaming"]:::svc
               RPC3["StreamRunSteps · StreamRunEpisodes  server-streaming"]:::svc
               RPC4["PublishRunSteps · PublishRunEpisodes  client-streaming"]:::svc
               RPC5["RegisterWorker · ControlStream  bidirectional"]:::svc
           end

           DM --> TS

           W0["Worker 0 · CleanRL / XuanCe · GPU 0"]:::worker
           W1["Worker 1 · CleanRL / XuanCe · GPU 1"]:::worker
           WN["Worker N · …"]:::worker

           TS -->|"subprocess"| W0 & W1 & WN
           W0 & W1 & WN -->|"PublishRunSteps / gRPC"| TS
       end

       TC -->|"gRPC :50055"| WIRE
       WIRE -->|"gRPC :50055"| DM

**Client responsibilities:**

- Render the PyQt6 GUI and handle user interaction
- Maintain a persistent gRPC channel to the daemon
- Stream live metrics and episode telemetry to the Analytics panel
- Issue run-lifecycle commands (submit, cancel, pause, resume)

**Server (daemon) responsibilities:**

- Expose the gRPC ``TrainerService`` and ``VideoFrameService`` endpoints
- Manage run registry and lifecycle state machine
  (``INIT → HANDSHAKE → READY → EXECUTING → TERMINATED``)
- Spawn and supervise RL worker subprocesses
- Aggregate and relay telemetry from workers to connected GUI clients
- Persist telemetry to SQLite for replay and analytics

Transport Protocol
------------------

MOSAIC uses **gRPC over HTTP/2** with Protocol Buffers (protobuf) as the
serialisation format. The service contract is defined in:

.. code-block:: text

   gym_gui/services/trainer/proto/trainer.proto

Key properties of the transport layer:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Property
     - Value
   * - Protocol
     - gRPC over HTTP/2
   * - Serialisation
     - Protocol Buffers v3
   * - Default port
     - ``50055``
   * - Default bind (server)
     - ``127.0.0.1:50055`` (loopback, single-machine mode)
   * - Remote bind (server)
     - ``0.0.0.0:50055`` (all interfaces, multi-machine mode)
   * - Transport security
     - Plaintext (trusted LAN); TLS provisioning required for WAN
   * - Streaming patterns
     - Unary, server-streaming, client-streaming, bidirectional

Deployment Modes
----------------

Single-Machine (Default)
^^^^^^^^^^^^^^^^^^^^^^^^^

Both the server and the GUI run on the same host. ``run.sh`` starts the
trainer daemon in the background, waits for it to be ready, then opens
the GUI — and shuts the daemon down when the GUI exits:

.. code-block:: bash

   ./run.sh

The server binds to ``127.0.0.1:50055`` and the GUI connects to the same
address. No network configuration is required.

Multi-Machine (Client-Server)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The GUI runs on a **client machine** (``hamid@HamidOnUbuntu`` — any machine
with a display) and the server runs on a **compute node**
(``Hamid@a1-R8428-G11`` — GPU server, lab machine, cloud VM).
Use ``run_server.sh`` on the compute node and ``run_client.sh`` on the
display machine.

.. list-table::
   :widths: 25 35 40
   :header-rows: 1

   * - Role
     - Typical hardware
     - Requirements
   * - **Client**
     - Laptop, desktop, workstation
     - Display (X11 / Wayland), Python 3.10–3.12, MOSAIC core
   * - **Server**
     - Multi-GPU lab server, HPC node, cloud GPU instance
     - CUDA-capable GPU(s), MOSAIC core + workers, no display needed

.. important::

   The MOSAIC GUI is a **native Qt application** that requires a display
   server. It cannot run on a headless compute node. X11 forwarding over
   SSH (``ssh -X``) is technically possible but produces unacceptable
   latency for real-time rendering. The recommended approach is to run the
   GUI locally and connect it to a remote daemon.

Server Setup
------------

1. Install MOSAIC on the Compute Server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install the core package and whichever worker libraries the server will run.
No GUI dependencies are required on the server side.

.. code-block:: bash

   git clone https://github.com/Abdulhamid97Mousa/MOSAIC.git
   cd MOSAIC

   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip setuptools wheel

   # Core package (includes gRPC daemon, no PyQt6 pull-in on headless)
   pip install -e .

   # Add worker libraries as needed
   pip install -e ".[cleanrl]"     # CleanRL (PPO, DQN, SAC, TD3)
   pip install -e ".[xuance]"      # XuanCe (MAPPO, QMIX, MADDPG)
   pip install -e ".[ray-rllib]"   # Ray RLlib (distributed training)

   # Compile gRPC stubs (must match client version)
   bash tools/generate_protos.sh

2. Start the Server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``run_server.sh``, which binds the daemon to all network interfaces
and logs to ``var/logs/trainer_daemon.log``:

.. code-block:: bash

   # Recommended: run inside a persistent terminal session (screen or tmux)
   screen -S mosaic-server
   source .venv/bin/activate
   bash run_server.sh

Expected output:

.. code-block:: text

   Starting MOSAIC server on 0.0.0.0:50055 ...
   Trainer gRPC server listening  {"listen": "0.0.0.0:50055"}

Detach from the screen session with ``Ctrl-A D``. Reconnect later with
``screen -r mosaic-server``.

**Custom bind address or port:**

.. code-block:: bash

   DAEMON_HOST=0.0.0.0 DAEMON_PORT=50056 bash run_server.sh

**With vLLM inference server (optional):**

.. code-block:: bash

   MOSAIC_START_VLLM=1 \
   MOSAIC_VLLM_MODEL=/path/to/Qwen2.5-3B-Instruct \
   bash run_server.sh

**Verify the daemon is listening:**

.. code-block:: bash

   # On the server
   ss -tlnp | grep 50055

Expected:

.. code-block:: text

   LISTEN  0  4096  *:50055  *:*  users:(("python",pid=...,fd=22))

Client Setup
------------

1. Install MOSAIC on the Client Machine
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The client requires only the core MOSAIC package — no worker libraries,
no GPU, no CUDA. The minimal installation is sufficient:

.. code-block:: bash

   git clone https://github.com/Abdulhamid97Mousa/MOSAIC.git
   cd MOSAIC

   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -e .

   bash tools/generate_protos.sh

.. list-table:: Installation requirements by role
   :widths: 40 30 30
   :header-rows: 1

   * - Component
     - Client
     - Server
   * - ``pip install -e .`` (core)
     - Required
     - Required
   * - gRPC stubs (``generate_protos.sh``)
     - Required
     - Required
   * - PyQt6 / display libraries
     - Required (installed by core)
     - Not required
   * - RL worker libraries (CleanRL, XuanCe, Ray)
     - Not required
     - Required
   * - CUDA / GPU drivers
     - Optional
     - Required for GPU training
   * - Display server (X11 / Wayland)
     - Required
     - Not required

2. Configure the Server Target
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set ``MOSAIC_DAEMON_TARGET`` to the server machine's IP address and port.
This variable is read by both ``run_client.sh`` and ``gym_gui.app`` at
startup, overriding the default ``127.0.0.1:50055``.

**Option A — Per-session (inline):**

.. code-block:: bash

   MOSAIC_DAEMON_TARGET=192.168.0.4:50055 bash run_client.sh

**Option B — Persistent (recommended):**

Add the variable to your local ``.env`` file so it applies automatically
every time you run MOSAIC:

.. code-block:: bash

   # In your local MOSAIC/.env
   MOSAIC_DAEMON_TARGET=192.168.0.4:50055

Then launch the client:

.. code-block:: bash

   bash run_client.sh

3. Launch the Client
^^^^^^^^^^^^^^^^^^^^^

Use ``run_client.sh`` to open the GUI and connect to the remote server.
It verifies the server is reachable before opening any window:

.. code-block:: bash

   MOSAIC_DAEMON_TARGET=192.168.0.4:50055 bash run_client.sh

Expected output:

.. code-block:: text

   Connecting to MOSAIC server at 192.168.0.4:50055 ...
   Server reachable at 192.168.0.4:50055
   Launching MOSAIC...

The GUI opens on the local display. All run submissions, telemetry
streaming, and worker management are handled transparently by the remote
server over gRPC.

.. tip::

   Add ``MOSAIC_DAEMON_TARGET=192.168.0.4:50055`` to your local ``.env``
   so you can just run ``bash run_client.sh`` without the prefix every time.

Verifying Connectivity
-----------------------

Before launching the GUI, confirm that the client can reach the server's
gRPC endpoint:

.. code-block:: bash

   # On the client machine
   source .venv/bin/activate
   python - << 'PY'
   import asyncio, grpc, sys

   SERVER = "192.168.0.4:50055"

   async def probe():
       channel = grpc.aio.insecure_channel(SERVER)
       try:
           await asyncio.wait_for(channel.channel_ready(), timeout=5.0)
           print(f"[OK]  gRPC channel ready — {SERVER}")
       except asyncio.TimeoutError:
           print(f"[ERR] Timeout — daemon not reachable at {SERVER}", file=sys.stderr)
           sys.exit(1)
       finally:
           await channel.close()

   asyncio.run(probe())
   PY

Running the Daemon as a System Service
---------------------------------------

For servers where the daemon should start automatically on boot, register
it as a ``systemd`` unit:

.. code-block:: ini

   # /etc/systemd/system/mosaic-daemon.service
   [Unit]
   Description=MOSAIC Trainer Daemon
   After=network-online.target
   Wants=network-online.target

   [Service]
   Type=simple
   User=<your-username>
   WorkingDirectory=/path/to/MOSAIC
   # Equivalent to: bash run_server.sh
   ExecStart=/path/to/MOSAIC/.venv/bin/python \
       -m gym_gui.services.trainer_daemon \
       --listen 0.0.0.0:50055
   Restart=on-failure
   RestartSec=10
   StandardOutput=journal
   StandardError=journal

   [Install]
   WantedBy=multi-user.target

.. code-block:: bash

   sudo systemctl daemon-reload
   sudo systemctl enable mosaic-daemon
   sudo systemctl start mosaic-daemon

   # Check status
   sudo systemctl status mosaic-daemon

   # Follow logs
   journalctl -u mosaic-daemon -f

Security Considerations
------------------------

.. warning::

   The default deployment uses **plaintext gRPC** (no TLS, no
   authentication). This is acceptable on a trusted local network
   (private lab LAN, VPN subnet) but must not be exposed to the public
   internet without additional hardening.

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Scenario
     - Recommended approach
   * - Trusted LAN (lab network)
     - Plaintext gRPC — default, no configuration required
   * - Public internet / cloud
     - SSH port-forward: ``ssh -L 50055:localhost:50055 user@server``,
       then keep ``MOSAIC_DAEMON_TARGET=127.0.0.1:50055`` on the client
   * - Production / multi-user
     - Provision TLS certificates and switch to ``add_secure_port`` in
       ``trainer_daemon.py``; add token-based authentication

**Server Persistence:**

The ``run_server.sh`` script starts the MOSAIC server with full detachment from the terminal:

* The daemon survives SSH disconnects using ``nohup`` and ``setsid``
* Process PID is saved to ``var/trainer/trainer.pid``
* Logs are written to ``var/logs/trainer_daemon.log``
* The server will continue running even after you log out

**Restrict daemon access to a known client IP via firewall:**

.. code-block:: bash

   sudo ufw allow from 192.168.0.x to any port 50055
   sudo ufw deny 50055

Server Management
~~~~~~~~~~~~~~~

Manage the detached server process using these commands:

.. code-block:: bash

   # Check if server is running (on server machine)
   ./run_server.sh status

   # Stop the server
   ./run_server.sh stop

   # View server logs
   tail -f var/logs/trainer_daemon.log

Troubleshooting
---------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Symptom
     - Resolution
   * - ``Cannot reach remote daemon``
     - Verify daemon is running: ``pgrep -af trainer_daemon``; check
       firewall: ``sudo ufw status``; confirm port is open on all
       interfaces: ``ss -tlnp | grep 50055``
   * - GUI opens then is immediately killed
     - Check available RAM: ``free -h``. The OS OOM killer terminates the
       process when free memory is exhausted. Close other applications.
   * - ``StatusCode.UNIMPLEMENTED`` or proto errors
     - Proto version mismatch. Run ``bash tools/generate_protos.sh`` on
       **both** client and server, then restart the daemon and GUI.
   * - ``ModuleNotFoundError: No module named 'dotenv'``
     - The system Python was used instead of the venv. Run
       ``source .venv/bin/activate`` before launching.
   * - Daemon binds but client cannot connect
     - Daemon may be bound to loopback only. Verify with
       ``ss -tlnp | grep 50055`` — the address column must show ``*:50055``
       (all interfaces), not ``127.0.0.1:50055``. Restart with
       ``bash run_server.sh`` which defaults to ``0.0.0.0``.

Next Steps
----------

- :doc:`../quickstart` — submit your first training run through the remote server
- :doc:`ubuntu` — full installation reference for Ubuntu
- :doc:`../../architecture/overview` — detailed description of MOSAIC's internal design

.. rubric:: Quick Reference

.. code-block:: bash

   # Single machine (development):
   bash run.sh

   # Split deployment — on the GPU server (Hamid@a1-R8428-G11):
   bash run_server.sh                    # Start server
   bash run_server.sh status              # Check if running
   bash run_server.sh stop                # Stop server

   # Split deployment — on the display machine (hamid@HamidOnUbuntu):
   MOSAIC_DAEMON_TARGET=<server-ip>:50055 bash run_client.sh
