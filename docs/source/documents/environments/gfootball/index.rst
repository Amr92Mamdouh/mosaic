GFootball (Google Research Football)
===================================

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="https://github.com/user-attachments/assets/df505885-525f-4f21-bacc-96f5464cf845?raw=true" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <p style="text-align:center; font-size:0.95em; color:#555; margin-top:6px;">
     <strong>Google Research Football running inside MOSAIC</strong>: human-controlled player, 10 Hz real-time stepping, RGB render view.
   </p>


Physics-based 3D football (soccer) simulation for single- and multi-agent
reinforcement learning research, developed by the Google Brain team.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **Paradigm**
     - Single-agent and multi-agent (simultaneous)
   * - **Stepping**
     - ``SINGLE_AGENT`` / ``SIMULTANEOUS``
   * - **Render**
     - ``RGB_ARRAY``
   * - **Adapter**
     - ``gym_gui/core/adapters/gfootball.py``


Unlike most MOSAIC families, GRF does **not** use ``gymnasium.make()``.  The
adapter calls ``gfootball.env.create_environment()`` directly and bridges the
legacy 4-tuple Gym API onto the Gymnasium 5-tuple contract used by MOSAIC.

Installation
------------

Step 1: System dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GRF engine is a C++ codebase compiled at install time.  On Debian/Ubuntu
install the toolchain and libraries first:

.. code-block:: bash

   sudo apt-get install -y \
       cmake g++ swig \
       libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev \
       libboost-all-dev libdirectfb-dev libst-dev timidity libgl1-mesa-dev

Step 2: Install the GRF Python package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MOSAIC ships GRF as a vendored checkout under ``3rd_party/environments/football/``.
Install it as an editable package:

.. code-block:: bash

   pip install -e 3rd_party/environments/football/

.. note::

   The first build compiles the full engine and typically takes 10-25 minutes.

Step 3: Runtime configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The GRF C++ engine always creates a top-level "Google Research Football"
window. Set ``GFOOTBALL_HIDDEN_WINDOW=1`` in your ``.env`` to create the window
with ``SDL_WINDOW_HIDDEN`` so it is never mapped; the OpenGL context and
``rgb_array`` frames are unchanged. Set it to ``0`` to restore the upstream
behaviour.

Single-Agent Environment List
-----------------------------

Short curriculum-style academy tasks that isolate individual football skills.
One agent controls a single player; the rest of the team is scripted.

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - Environment ID
     - Description
   * - ``academy_empty_goal_close``
     - Score from just outside the goal area (no keeper)
   * - ``academy_empty_goal``
     - Dribble from midfield into an empty goal
   * - ``academy_run_to_score``
     - Run past static defenders and score
   * - ``academy_run_to_score_with_keeper``
     - Run past static defenders and score against an active goalkeeper
   * - ``academy_pass_and_shoot_with_keeper``
     - Pass to a teammate, then shoot past the keeper
   * - ``academy_run_pass_and_shoot_with_keeper``
     - Run, receive a pass, and shoot against a keeper
   * - ``academy_3_vs_1_with_keeper``
     - Three attackers vs one defender and a keeper (overload drill)
   * - ``academy_corner``
     - Score from a corner kick situation
   * - ``academy_counterattack_easy``
     - Counterattack with a numerical advantage (easy defence)
   * - ``academy_counterattack_hard``
     - Counterattack against a stronger, organised defence
   * - ``academy_single_goal_versus_lazy``
     - Full team vs a passive (non-defending) opposition

Multi-Agent Environment List
----------------------------

Full-match and reduced-player scenarios where multiple agents are controlled simultaneously.
Each controlled player is an independent agent sharing the same discrete action space.

.. list-table::
   :header-rows: 1
   :widths: 34 14 52

   * - Environment ID
     - Agents
     - Description
   * - ``1_vs_1_easy``
     - 1
     - One attacker vs one defender (minimal multi-agent setting)
   * - ``5_vs_5``
     - 5
     - Five-a-side match (5 controlled + 5 built-in AI opponents)
   * - ``11_vs_11_easy_stochastic``
     - 11
     - Full 11-vs-11 match against an easy built-in AI
   * - ``11_vs_11_stochastic``
     - 11
     - Full 11-vs-11 match at default difficulty
   * - ``11_vs_11_hard_stochastic``
     - 11
     - Full 11-vs-11 match against a hard built-in AI

Action Space
------------

19 discrete actions (``default`` action set). Sprint and dribble are **sticky**; they persist
across steps until explicitly released.

.. raw:: html

   <table style="width:100%; border-collapse: collapse; margin: 10px 0;">
     <tr style="background-color: #f0f0f0;">
       <th style="border: 1px solid #ddd; padding: 6px;">Index</th>
       <th style="border: 1px solid #ddd; padding: 6px;">Action</th>
       <th style="border: 1px solid #ddd; padding: 6px;">Key</th>
       <th style="border: 1px solid #ddd; padding: 6px;">Notes</th>
     </tr>
     <tr><td style="border:1px solid #ddd;padding:6px;">0</td><td style="border:1px solid #ddd;padding:6px;">idle</td><td style="border:1px solid #ddd;padding:6px;">0</td><td style="border:1px solid #ddd;padding:6px;">No action</td></tr>
     <tr><td style="border:1px solid #ddd;padding:6px;">1</td><td style="border:1px solid #ddd;padding:6px;">left</td><td style="border:1px solid #ddd;padding:6px;">A</td><td style="border:1px solid #ddd;padding:6px;">Move left</td></tr>
     <tr><td style="border:1px solid #ddd;padding:6px;">2</td><td style="border:1px solid #ddd;padding:6px;">top_left</td><td style="border:1px solid #ddd;padding:6px;">Q</td><td style="border:1px solid #ddd;padding:6px;">Move diagonally</td></tr>
     <tr><td style="border:1px solid #ddd;padding:6px;">3</td><td style="border:1px solid #ddd;padding:6px;">top</td><td style="border:1px solid #ddd;padding:6px;">W</td><td style="border:1px solid #ddd;padding:6px;">Move up</td></tr>
     <tr><td style="border:1px solid #ddd;padding:6px;">4</td><td style="border:1px solid #ddd;padding:6px;">top_right</td><td style="border:1px solid #ddd;padding:6px;">E</td><td style="border:1px solid #ddd;padding:6px;">Move diagonally</td></tr>
     <tr><td style="border:1px solid #ddd;padding:6px;">5</td><td style="border:1px solid #ddd;padding:6px;">right</td><td style="border:1px solid #ddd;padding:6px;">D</td><td style="border:1px solid #ddd;padding:6px;">Move right</td></tr>
     <tr><td style="border:1px solid #ddd;padding:6px;">6</td><td style="border:1px solid #ddd;padding:6px;">bottom_right</td><td style="border:1px solid #ddd;padding:6px;">C</td><td style="border:1px solid #ddd;padding:6px;">Move diagonally</td></tr>
     <tr><td style="border:1px solid #ddd;padding:6px;">7</td><td style="border:1px solid #ddd;padding:6px;">bottom</td><td style="border:1px solid #ddd;padding:6px;">S</td><td style="border:1px solid #ddd;padding:6px;">Move down</td></tr>
     <tr><td style="border:1px solid #ddd;padding:6px;">8</td><td style="border:1px solid #ddd;padding:6px;">bottom_left</td><td style="border:1px solid #ddd;padding:6px;">Z</td><td style="border:1px solid #ddd;padding:6px;">Move diagonally</td></tr>
     <tr><td style="border:1px solid #ddd;padding:6px;">9</td><td style="border:1px solid #ddd;padding:6px;">long_pass</td><td style="border:1px solid #ddd;padding:6px;">J</td><td style="border:1px solid #ddd;padding:6px;">Long through ball</td></tr>
     <tr><td style="border:1px solid #ddd;padding:6px;">10</td><td style="border:1px solid #ddd;padding:6px;">high_pass</td><td style="border:1px solid #ddd;padding:6px;">K</td><td style="border:1px solid #ddd;padding:6px;">Lofted pass</td></tr>
     <tr><td style="border:1px solid #ddd;padding:6px;">11</td><td style="border:1px solid #ddd;padding:6px;">short_pass</td><td style="border:1px solid #ddd;padding:6px;">L</td><td style="border:1px solid #ddd;padding:6px;">Ground pass</td></tr>
     <tr><td style="border:1px solid #ddd;padding:6px;">12</td><td style="border:1px solid #ddd;padding:6px;">shot</td><td style="border:1px solid #ddd;padding:6px;">Space</td><td style="border:1px solid #ddd;padding:6px;">Shoot at goal</td></tr>
     <tr><td style="border:1px solid #ddd;padding:6px;">13</td><td style="border:1px solid #ddd;padding:6px;">sprint</td><td style="border:1px solid #ddd;padding:6px;">Shift</td><td style="border:1px solid #ddd;padding:6px;">Toggle sprint (sticky)</td></tr>
     <tr><td style="border:1px solid #ddd;padding:6px;">14</td><td style="border:1px solid #ddd;padding:6px;">release_direction</td><td style="border:1px solid #ddd;padding:6px;">X</td><td style="border:1px solid #ddd;padding:6px;">Stop moving</td></tr>
     <tr><td style="border:1px solid #ddd;padding:6px;">15</td><td style="border:1px solid #ddd;padding:6px;">release_sprint</td><td style="border:1px solid #ddd;padding:6px;">V</td><td style="border:1px solid #ddd;padding:6px;">Stop sprinting</td></tr>
     <tr><td style="border:1px solid #ddd;padding:6px;">16</td><td style="border:1px solid #ddd;padding:6px;">sliding</td><td style="border:1px solid #ddd;padding:6px;">T</td><td style="border:1px solid #ddd;padding:6px;">Sliding tackle (defence only)</td></tr>
     <tr><td style="border:1px solid #ddd;padding:6px;">17</td><td style="border:1px solid #ddd;padding:6px;">dribble</td><td style="border:1px solid #ddd;padding:6px;">R</td><td style="border:1px solid #ddd;padding:6px;">Toggle dribble (sticky)</td></tr>
     <tr><td style="border:1px solid #ddd;padding:6px;">18</td><td style="border:1px solid #ddd;padding:6px;">release_dribble</td><td style="border:1px solid #ddd;padding:6px;">F</td><td style="border:1px solid #ddd;padding:6px;">Stop dribbling</td></tr>
   </table>

Observation Space
-----------------

.. list-table::
   :header-rows: 1
   :widths: 22 12 66

   * - Representation
     - Shape
     - Description
   * - ``simple115v2`` (default)
     - (115,)
     - Fixed vector: ball position/direction (6), left team pos/dir (22), right team pos/dir (22), active player one-hot (11), sticky actions (10), game mode one-hot (7), ball ownership, score delta
   * - ``extracted``
     - (72, 96, 4)
     - Super minimap: 4-channel image encoding player positions, ball, and active player on a 72×96 grid
   * - ``pixels``
     - (72, 96, 3)
     - Downscaled RGB frame (requires ``render=True``)
   * - ``raw``
     - dict
     - Full game state dictionary with all player/ball info

Reward Structure
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Description
   * - ``scoring`` (default)
     - +1 for scoring a goal, −1 for conceding. Sparse signal.
   * - ``checkpoints``
     - Dense reward: small positive signal when the ball crosses distance checkpoints toward the opponent goal. Aids early training.
   * - ``scoring,checkpoints``
     - Combined: both goal reward and checkpoint distance reward.

Configuration
-------------

``GFootballConfig`` (``gym_gui/config/game_configs.py``):

.. list-table::
   :header-rows: 1
   :widths: 34 16 50

   * - Field
     - Default
     - Description
   * - ``representation``
     - ``simple115v2``
     - ``simple115``, ``simple115v2``, ``extracted``, ``pixels``,
       ``pixels_gray``, or ``raw``
   * - ``rewards``
     - ``scoring``
     - ``scoring`` and/or ``checkpoints`` (comma-separated)
   * - ``number_of_left_players_agent_controls``
     - ``1``
     - Left-team players under agent control (0-11)
   * - ``number_of_right_players_agent_controls``
     - ``0``
     - Right-team players under agent control (0-11)
   * - ``stacked``
     - ``False``
     - Stack 4 consecutive frames (pixel/extracted only)
   * - ``action_set``
     - ``default``
     - ``default`` (19), ``v2`` (20), or ``full`` (33) actions
   * - ``render``
     - ``True``
     - Required for the MOSAIC Render View
   * - ``max_steps``
     - ``3000``
     - Maximum steps per episode
   * - ``seed``
     - ``None``
     - Random seed

Known Issues and Limitations
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Issue
     - Detail / workaround
   * - Boost.Python / Python ABI mismatch
     - The engine links Boost.Python, which is built for one specific Python
       minor version.  Ubuntu 22.04 only ships ``libboost_python310`` while
       MOSAIC runs Python 3.11, so the engine imports with
       ``SystemError: initialization of _gameplayfootball raised unreported
       exception``.  Build Boost 1.82 with ``python=3.11`` and configure the
       engine with ``-DMOSAIC_PYTHON_OVERRIDE=ON``.  See
       ``var/data/gfootball/README.md``.
   * - CMake 3.22 cannot find Python 3.11
     - The bundled ``FindPython`` module caps out at 3.10.  The vendored
       ``CMakeLists.txt`` accepts ``-DMOSAIC_PYTHON_OVERRIDE=ON`` so the
       interpreter include and library paths can be supplied directly.
   * - Editable install fails on modern setuptools
     - ``setup.py``'s ``copy_fonts()`` raises
       ``[Errno 20] Not a directory: .../gfootball_engine/fonts`` under the
       setuptools ``editable_wheel`` backend.  Build the engine directly, then
       create the ``gfootball_engine`` symlink, copy ``third_party/fonts``, and
       add a ``.pth`` path entry.
   * - Missing fonts crash the engine
     - Without ``gfootball_engine/fonts/AlegreyaSansSC-ExtraBold.ttf`` the
       engine aborts with a fatal error and SIGSEGV during
       ``create_environment()``.
   * - Gym 0.26 breaks GRF wrappers
     - GRF targets ``gym<=0.21``.  Under gym 0.26 the ``Wrapper``,
       ``ObservationWrapper``, and ``RewardWrapper`` base classes use the
       5-tuple API, producing ``ValueError: not enough values to unpack
       (expected 2, got 1)``.  ``gfootball/env/wrappers.py`` carries a MOSAIC
       compatibility shim that restores the legacy API.
   * - Long compile time
     - The C++ engine is built from source on first install.  Set
       ``GFOOTBALL_USE_PREBUILT_SO=1`` to use the bundled prebuilt shared
       object, but note it is only compatible with matching Python and glibc
       versions.
   * - ``build_game_engine.sh`` uses system ``python3``
     - The build script probes ``psutil`` via the system interpreter to pick a
       parallelism level.  If the system ``psutil`` is broken the build aborts.
       Run the build with the MOSAIC virtual environment first on ``PATH``.
   * - Legacy Gym API
     - GRF targets ``gym<=0.21``.  ``step()`` returns a 4-tuple and ``reset()``
       returns observations only.  The MOSAIC adapter performs the conversion;
       do not call the underlying environment directly.
   * - Action and observation spaces
     - GRF builds ``gym.spaces`` objects.  The adapter constructs equivalent
       ``gymnasium.spaces`` so base-class contracts hold.
   * - Frames require ``render=True``
     - With ``render=False`` no RGB frame is produced and the Render View stays
       blank.  ``GFootballConfig`` defaults to ``True``.
   * - Multi-agent rewards are arrays
     - GRF returns one reward per controlled player.  The adapter reports the
       sum as the scalar step reward and exposes the full vector under
       ``info["agent_rewards"]``.
   * - Headless machines
     - Rendering needs a GL context.  On headless hosts run under ``xvfb-run``
       or an equivalent virtual display.
   * - Engine opens its own SDL window
     - The GRF C++ engine always creates a top-level "Google Research Football"
       window, which appears alongside the MOSAIC Qt Shell. MOSAIC patches the
       renderer to honour ``GFOOTBALL_HIDDEN_WINDOW=1`` (set in ``.env``), which
       creates the window with ``SDL_WINDOW_HIDDEN``: the OpenGL context and the
       ``rgb_array`` frames are unchanged, but the window is never mapped. Set
       it to ``0`` to restore the upstream behaviour.

MAPPO Training via XuanCe
-------------------------

Training RL agents (MAPPO via XuanCe) on GRF is fully working inside MOSAIC.
The video below shows a live MAPPO training session on the
``academy_3_vs_1_with_keeper`` scenario:

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="https://github.com/user-attachments/assets/118bbc56-2ca0-4c7b-8c0c-c24464dbad43?raw=true" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <p style="text-align:center; font-size:0.95em; color:#555; margin-top:6px;">
     <strong>MAPPO training on GRF</strong> (scenario <code>academy_3_vs_1_with_keeper</code>) via <code>xuance_worker</code> inside MOSAIC.
   </p>

Hyperparameters
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Setting
     - Value
   * - Algorithm
     - MAPPO (parameter-sharing)
   * - Environment / scenario
     - ``Football`` / ``academy_3_vs_1_with_keeper``
   * - Observation representation
     - ``simple115v2``
   * - Reward shaping
     - ``scoring,checkpoints``
   * - Number of controlled agents
     - 3
   * - Seed
     - 1
   * - Parallel environments
     - 50
   * - Rollout buffer size
     - 400
   * - Learning rate
     - 5.0e-4
   * - PPO epochs / minibatches
     - 15 / 2
   * - Discount γ / GAE λ
     - 0.99 / 0.95
   * - Clip range
     - 0.2
   * - Episode length
     - 200
   * - Total training steps
     - 25,000,000
   * - Recurrent policy
     - GRU, 1 layer, hidden size 64
   * - Device
     - cuda:0

Full config:
``3rd_party/workers/xuance_worker/xuance/examples/mappo/mappo_football_configs/3v1.yaml``

Citation
--------

.. code-block:: bibtex

   @inproceedings{kurach2020google,
     title     = {Google Research Football: A Novel Reinforcement Learning Environment},
     author    = {Kurach, Karol and Raichuk, Anton and Sta{\'n}czyk, Piotr and
                  Zajac, Micha{\l} and Bachem, Olivier and Espeholt, Lasse and
                  Riquelme, Carlos and Vincent, Damien and Michalski, Marcin and
                  Bousquet, Olivier and Gelly, Sylvain},
     booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
     year      = {2020},
   }
