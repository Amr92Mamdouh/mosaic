SMAC
====

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="https://github.com/user-attachments/assets/6c1bde96-c0ab-401c-ba80-56cdf3ab6807?raw=true" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <p style="text-align:center; font-size:0.95em; color:#555; margin-top:6px;">
     <strong>SMAC running inside MOSAIC</strong>: cooperative micromanagement scenario with per-step action masking via the Operator interface.
   </p>

The StarCraft Multi-Agent Challenge (SMAC) is WhiRL's environment for research on
cooperative MARL algorithms. SMAC uses **StarCraft II**, a real-time strategy game
developed by Blizzard Entertainment, as its underlying engine, and ships 23
hand-designed cooperative micromanagement maps.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **GitHub**
     - https://github.com/oxwhirl/smac
   * - **Paradigm**
     - Multi-agent (simultaneous)
   * - **Stepping**
     - ``SIMULTANEOUS``
   * - **Adapter**
     - ``gym_gui/core/adapters/smac.py``

Installation
------------

Step 1: Install the SMAC Python package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MOSAIC ships SMAC as a vendored checkout under ``3rd_party/environments/smac/``.
Install it as an editable package so the venv resolves ``import smac``:

.. code-block:: bash

   pip install -e 3rd_party/environments/smac/

Alternatively, install directly from the upstream GitHub (not the pattern used
in-repo):

.. code-block:: bash

   pip install git+https://github.com/oxwhirl/smac.git

Or clone and install with its dependencies:

.. code-block:: bash

   git clone https://github.com/oxwhirl/smac.git
   cd smac/
   pip install -e .

Step 2: Install StarCraft II
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Linux.** Download the headless Linux client from `Blizzard's repository
<https://github.com/Blizzard/s2client-proto#downloads>`_ and extract it to
``var/data/StarCraftII`` inside the repository. The archive ships with a
top-level ``StarCraftII/`` directory, so extracting under ``var/data/`` produces
the correct final path automatically.

**Windows / macOS.** Install StarCraft II from `BATTLE.NET
<https://www.battle.net>`_ or `starcraft2.blizzard.com
<https://starcraft2.blizzard.com>`_, then point ``SC2PATH`` at the install
directory.

.. note::

   ``SC2PATH`` is set by MOSAIC's ``.env`` to ``$PWD/var/data/StarCraftII``
   (project-local headless client, not committed to git). ``$PWD`` expands to
   the mosaic repo root when ``.env`` is sourced by ``run.sh``. If SC2 lives
   elsewhere on your machine, override ``SC2PATH`` in your local ``.env``.
   MOSAIC resolves the path in this order (see ``gym_gui/core/adapters/smac.py``):
   ``config.sc2_path`` → ``SC2PATH`` env var → ``~/StarCraftII`` (pysc2 default).

Step 3: SMAC Maps
~~~~~~~~~~~~~~~~~

Once you have installed SMAC and StarCraft II, download the SMAC Maps and
extract them into ``$SC2PATH/Maps``. If you installed SMAC via git, simply copy
the ``SMAC_Maps`` directory from ``smac/env/starcraft2/maps/`` into
``$SC2PATH/Maps``:

.. code-block:: bash

   cp -r 3rd_party/environments/smac/smac/env/starcraft2/maps/SMAC_Maps \
         "$SC2PATH/Maps/"

.. note::

   When using custom maps created with the SC2 Editor, you need to modify the
   ``version`` in ``t3Terrain.xml`` to ``114`` using ``mpqeditor``. Refer to the
   upstream `issue on GitHub <https://github.com/oxwhirl/smac/issues>`_ for
   details.

All 23 Maps
-----------

.. list-table::
   :header-rows: 1
   :widths: 22 14 64

   * - Map
     - Difficulty
     - Description
   * - ``3m``
     - Easy
     - 3 Marines vs 3 Marines (symmetric entry-level scenario)
   * - ``8m``
     - Easy
     - 8 Marines vs 8 Marines (symmetric, larger scale)
   * - ``2s3z``
     - Easy
     - 2 Stalkers + 3 Zealots vs same (mixed ranged/melee)
   * - ``3s5z``
     - Easy
     - 3 Stalkers + 5 Zealots vs same
   * - ``1c3s5z``
     - Easy
     - 1 Colossus + 3 Stalkers + 5 Zealots vs same (highly heterogeneous)
   * - ``2m_vs_1z``
     - Easy
     - 2 Marines vs 1 Zealot (asymmetric forces)
   * - ``2s_vs_1sc``
     - Easy
     - 2 Stalkers vs 1 Spine Crawler (static defender)
   * - ``25m``
     - Easy
     - 25 Marines vs 25 Marines (large-scale homogeneous)
   * - ``3s_vs_3z``
     - Hard
     - 3 Stalkers vs 3 Zealots (range advantage vs durability)
   * - ``3s_vs_4z``
     - Hard
     - 3 Stalkers vs 4 Zealots (asymmetric numbers)
   * - ``3s_vs_5z``
     - Hard
     - 3 Stalkers vs 5 Zealots (significant numerical disadvantage)
   * - ``5m_vs_6m``
     - Hard
     - 5 Marines vs 6 Marines (asymmetric numbers)
   * - ``8m_vs_9m``
     - Hard
     - 8 Marines vs 9 Marines (asymmetric, medium scale)
   * - ``10m_vs_11m``
     - Hard
     - 10 Marines vs 11 Marines (asymmetric, large scale)
   * - ``bane_vs_bane``
     - Hard
     - Banelings vs Banelings (fragile units, explosion chain mechanic)
   * - ``2c_vs_64zg``
     - Hard
     - 2 Colossi vs 64 Zerglings (massive numerical asymmetry, AoE required)
   * - ``MMM``
     - Hard
     - 1 Medivac + 2 Marauders + 7 Marines vs same (hetero, no healer on enemy)
   * - ``corridor``
     - Hard
     - 6 Zealots vs 24 Zerglings in a narrow corridor
   * - ``27m_vs_30m``
     - Hard
     - 27 Marines vs 30 Marines (large-scale asymmetric)
   * - ``3s5z_vs_3s6z``
     - Super Hard
     - 3 Stalkers + 5 Zealots vs 3 Stalkers + 6 Zealots (one extra Zealot)
   * - ``6h_vs_8z``
     - Super Hard
     - 6 Hydralisks vs 8 Zealots (heavily armoured melee opposition)
   * - ``so_many_baneling``
     - Super Hard
     - 7 Marines vs 32 Banelings (high-explosion-chain pressure)
   * - ``MMM2``
     - Super Hard
     - 1 Medivac + 2 Marauders + 7 Marines vs 1 Medivac + 3 Marauders + 8 Marines

Action Space
------------

.. raw:: html

   <table style="width:100%; border-collapse: collapse; margin: 10px 0;">
     <tr style="background-color: #f0f0f0;">
       <th style="border: 1px solid #ddd; padding: 6px;">Index</th>
       <th style="border: 1px solid #ddd; padding: 6px;">Action</th>
       <th style="border: 1px solid #ddd; padding: 6px;">Notes</th>
     </tr>
     <tr><td style="border: 1px solid #ddd; padding: 6px;">0</td><td style="border: 1px solid #ddd; padding: 6px;">NO-OP</td><td style="border: 1px solid #ddd; padding: 6px;">Dead agents only</td></tr>
     <tr><td style="border: 1px solid #ddd; padding: 6px;">1</td><td style="border: 1px solid #ddd; padding: 6px;">STOP</td><td style="border: 1px solid #ddd; padding: 6px;">Hold position</td></tr>
     <tr><td style="border: 1px solid #ddd; padding: 6px;">2</td><td style="border: 1px solid #ddd; padding: 6px;">MOVE NORTH</td><td style="border: 1px solid #ddd; padding: 6px;"></td></tr>
     <tr><td style="border: 1px solid #ddd; padding: 6px;">3</td><td style="border: 1px solid #ddd; padding: 6px;">MOVE SOUTH</td><td style="border: 1px solid #ddd; padding: 6px;"></td></tr>
     <tr><td style="border: 1px solid #ddd; padding: 6px;">4</td><td style="border: 1px solid #ddd; padding: 6px;">MOVE EAST</td><td style="border: 1px solid #ddd; padding: 6px;"></td></tr>
     <tr><td style="border: 1px solid #ddd; padding: 6px;">5</td><td style="border: 1px solid #ddd; padding: 6px;">MOVE WEST</td><td style="border: 1px solid #ddd; padding: 6px;"></td></tr>
     <tr><td style="border: 1px solid #ddd; padding: 6px;">6+</td><td style="border: 1px solid #ddd; padding: 6px;">ATTACK ENEMY i</td><td style="border: 1px solid #ddd; padding: 6px;">One per visible enemy in range</td></tr>
   </table>

**Action Masking:** ``get_avail_agent_actions()`` returns a binary mask per agent per step.
Dead agents can only NO-OP. Attack actions are only available for enemies within attack range.

Observation Space
-----------------

Each agent's local observation (units within sight range):

- **Allied units:** distance, relative x/y, health, shield, unit type
- **Enemy units:** distance, relative x/y, health, shield, unit type
- **Self:** own health, shield, unit type, own position (optional)
- **Optional:** pathing grid (terrain walkability), terrain height

**Global state** (available during training for CTDE critics): positions, health, shields
for ALL units on both teams.

Reward Structure
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Description
   * - **Shaped** (default)
     - Per-step reward based on damage dealt and received. +200 bonus for winning.
   * - **Sparse**
     - +1 for winning the battle, 0 otherwise.

All agents share the same team reward.

MAPPO Training via XuanCe
-------------------------

MAPPO training on SMAC via ``xuance_worker`` is fully working inside MOSAIC.
The video below shows a live MAPPO training session on the ``3m`` map:

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="https://github.com/user-attachments/assets/1f1b362d-0046-4393-ba23-675269239b3d?raw=true" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <p style="text-align:center; font-size:0.95em; color:#555; margin-top:6px;">
     <strong>MAPPO training on SMAC</strong> (map <code>3m</code>) via <code>xuance_worker</code> inside MOSAIC.
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
   * - Environment / map
     - ``StarCraft2`` / ``3m``
   * - Seed
     - 1
   * - Parallel environments
     - 4
   * - Rollout buffer size
     - 128
   * - Learning rate
     - 7.0e-4
   * - PPO epochs / minibatches
     - 15 / 1
   * - Discount γ / GAE λ
     - 0.99 / 0.95
   * - Clip range
     - 0.2
   * - Total training steps
     - 1,000,000
   * - Recurrent policy
     - GRU, 1 layer, hidden size 64
   * - Action masking
     - Enabled
   * - Device
     - cuda:0

Full config:
``3rd_party/workers/xuance_worker/xuance_worker/configs/mappo/smac/3m.yaml``

Citation
--------

.. code-block:: bibtex

   @article{samvelyan19smac,
     author       = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philip H. S. Torr and Jakob Foerster and Shimon Whiteson},
     title        = {The StarCraft Multi-Agent Challenge},
     journal      = {CoRR},
     volume       = {abs/1902.04043},
     year         = {2019},
   }
