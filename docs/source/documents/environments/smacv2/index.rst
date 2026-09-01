SMACv2
======

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="https://github.com/user-attachments/assets/dc83a7d3-8088-49f5-a1ad-05acece24b50?raw=true" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <p style="text-align:center; font-size:0.95em; color:#555; margin-top:6px;">
     <strong>SMACv2 running inside MOSAIC:</strong> procedural unit compositions randomised each episode; action masks forwarded to the Operator interface.
   </p>

SMACv2 is the second-generation StarCraft Multi-Agent Challenge, extending SMAC
with **procedural unit generation**: team compositions and start positions are
randomised at every episode reset, which makes memorisation-based policies far
less effective and produces a more faithful test of cooperative generalisation.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **GitHub**
     - https://github.com/oxwhirl/smacv2
   * - **Paradigm**
     - Multi-agent (simultaneous)
   * - **Stepping**
     - ``SIMULTANEOUS``
   * - **Adapter**
     - ``gym_gui/core/adapters/smacv2.py``

Installation
------------

Step 1: Install the SMACv2 Python package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MOSAIC ships SMACv2 as a vendored checkout under
``3rd_party/environments/smacv2/``. Install it as an editable package so the
venv resolves ``import smacv2``:

.. code-block:: bash

   pip install -e 3rd_party/environments/smacv2/

Alternatively, install directly from the upstream GitHub (not the pattern used
in-repo):

.. code-block:: bash

   pip install git+https://github.com/oxwhirl/smacv2.git

Or clone and install with its dependencies:

.. code-block:: bash

   git clone https://github.com/oxwhirl/smacv2.git
   cd smacv2/
   pip install -e .

.. note::

   ``smac`` and ``smacv2`` are genuinely separate packages and MOSAIC installs
   both side by side. Together they trigger a latent ``DuplicateMapError`` in
   ``pysc2.maps.lib.get_maps()`` because both packages register overlapping map
   names (``3m``, ``8m``, ``MMM2``, ...) into pysc2's global class registry.
   MOSAIC patches this at adapter-import time by making ``get_maps()`` tolerant
   of duplicates (first registration wins). See
   ``gym_gui/core/adapters/smac.py::_patch_pysc2_duplicate_map_tolerance``.

Step 2: Install StarCraft II
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same SC2 client as SMAC. Install once and reuse.

**Linux.** Download from `Blizzard's repository
<https://github.com/Blizzard/s2client-proto#downloads>`_ and extract into the
repository at ``var/data/StarCraftII``. The archive already contains a
top-level ``StarCraftII/`` directory, so extracting under ``var/data/`` gives
you the correct final path.

**Windows / macOS.** Install StarCraft II from `BATTLE.NET
<https://www.battle.net>`_ or `starcraft2.blizzard.com
<https://starcraft2.blizzard.com>`_, then either copy or symlink the install
into ``var/data/StarCraftII`` inside the repository, or point ``SC2PATH`` at
the install directory.

.. note::

   MOSAIC's SMAC adapter (``gym_gui/core/adapters/smac.py``) resolves the
   StarCraft II installation path in this order:

   1. ``config.sc2_path`` on the SMAC game panel, if set.
   2. The ``SC2PATH`` environment variable, if set.
   3. The in-repo fallback ``var/data/StarCraftII`` (the ``VAR_SC2_DIR``
      constant defined in ``gym_gui/config/paths.py``).

   The recommended setup is to install SC2 into ``var/data/StarCraftII``
   and set ``SC2PATH`` to that absolute path in ``.env`` (already done in the
   project's ``.env`` and ``.env.example``). pysc2 reads ``SC2PATH`` directly,
   so setting it explicitly avoids any ambiguity from the ``~/StarCraftII``
   default. If SC2 already lives elsewhere on your machine, override
   ``SC2PATH`` in your local ``.env``.

Step 3: SMACv2 Maps
~~~~~~~~~~~~~~~~~~~

SMACv2 ships four **procedural** map files (``10gen_terran.SC2Map``,
``10gen_protoss.SC2Map``, ``10gen_zerg.SC2Map``, ``10gen_empty.SC2Map``) in
addition to the SMAC map set. Copy them into your SC2 install:

.. code-block:: bash

   cp -r 3rd_party/environments/smacv2/smacv2/env/starcraft2/maps/SMAC_Maps \
         "$SC2PATH/Maps/"

Both SMAC and SMACv2 place their maps under the same ``SMAC_Maps`` directory,
so a single copy step covers both packages if you install them together.

.. note::

   When using custom maps created with the SC2 Editor, modify ``version`` in
   ``t3Terrain.xml`` to ``114`` using ``mpqeditor``.

All 15 Maps
-----------

Each map name encodes: ``<race>_<allies>_vs_<enemies>``. Compositions are randomised
from the race's unit pool every episode reset.

.. list-table::
   :header-rows: 1
   :widths: 30 15 15 40

   * - Map
     - Allies
     - Enemies
     - Unit Pool
   * - ``protoss_5_vs_5``
     - 5
     - 5
     - Stalkers, Zealots, Colossi
   * - ``protoss_10_vs_10``
     - 10
     - 10
     - Stalkers, Zealots, Colossi
   * - ``protoss_10_vs_11``
     - 10
     - 11
     - Stalkers, Zealots, Colossi
   * - ``protoss_20_vs_20``
     - 20
     - 20
     - Stalkers, Zealots, Colossi
   * - ``protoss_20_vs_23``
     - 20
     - 23
     - Stalkers, Zealots, Colossi
   * - ``terran_5_vs_5``
     - 5
     - 5
     - Marines, Marauders, Medivacs
   * - ``terran_10_vs_10``
     - 10
     - 10
     - Marines, Marauders, Medivacs
   * - ``terran_10_vs_11``
     - 10
     - 11
     - Marines, Marauders, Medivacs
   * - ``terran_20_vs_20``
     - 20
     - 20
     - Marines, Marauders, Medivacs
   * - ``terran_20_vs_23``
     - 20
     - 23
     - Marines, Marauders, Medivacs
   * - ``zerg_5_vs_5``
     - 5
     - 5
     - Zerglings, Banelings, Hydralisks
   * - ``zerg_10_vs_10``
     - 10
     - 10
     - Zerglings, Banelings, Hydralisks
   * - ``zerg_10_vs_11``
     - 10
     - 11
     - Zerglings, Banelings, Hydralisks
   * - ``zerg_20_vs_20``
     - 20
     - 20
     - Zerglings, Banelings, Hydralisks
   * - ``zerg_20_vs_23``
     - 20
     - 23
     - Zerglings, Banelings, Hydralisks

.. note::

   Unit start positions **and** type compositions are randomised at each ``env.reset()``.
   The adapter re-queries ``get_env_info()`` after every reset because agent count and
   observation/action shapes may change between episodes.

Action Space
------------

Identical to SMAC. Discrete per-agent action space with action masking:

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

Observation Space
-----------------

Same structure as SMAC, but shapes may vary between episodes:

- **Allied units:** distance, relative x/y, health, shield, unit type
- **Enemy units:** distance, relative x/y, health, shield, unit type
- **Self:** own health, shield, unit type, own position (optional)

**Global state** (available during training): positions, health, shields for ALL units.

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

MAPPO training on SMACv2 via ``xuance_worker`` is fully working inside MOSAIC.
The video below shows a live MAPPO training session on the asymmetric,
disadvantaged 20-vs-23 Terran scenario:

.. raw:: html

   <video style="width:100%; max-width:100%; height:auto; border-radius:8px; box-shadow:0 2px 8px rgba(0,0,0,0.15);" controls autoplay muted loop playsinline>
     <source src="https://github.com/user-attachments/assets/a752f29d-6b40-4a36-adea-21f3bc09069c?raw=true" type="video/mp4">
     Your browser does not support the video tag.
   </video>
   <p style="text-align:center; font-size:0.95em; color:#555; margin-top:6px;">
     <strong>MAPPO training on SMACv2</strong> (map <code>terran_20_vs_23</code>) via <code>xuance_worker</code> inside MOSAIC.
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
     - ``StarCraft2v2`` / ``terran_20_vs_23``
   * - Seed
     - 1
   * - Parallel environments
     - 4
   * - Rollout buffer size
     - 256
   * - Learning rate
     - 5.0e-4
   * - PPO epochs / minibatches
     - 15 / 2
   * - Discount γ / GAE λ
     - 0.99 / 0.95
   * - Clip range
     - 0.2
   * - Total training steps
     - 20,000,000
   * - Recurrent policy
     - GRU, 1 layer, hidden size 64
   * - Action masking
     - Enabled
   * - Device
     - cuda:0

Full config:
``3rd_party/workers/xuance_worker/xuance_worker/configs/mappo/smacv2/terran_20_vs_23.yaml``

Citation
--------

.. code-block:: bibtex

   @article{ellis2023smacv2,
     author       = {Benjamin Ellis and Jonathan Cook and Skander Moalla and Mikayel Samvelyan and Mingfei Sun and Anuj Mahajan and Jakob Foerster and Shimon Whiteson},
     title        = {SMACv2: An Improved Benchmark for Cooperative Multi-Agent Reinforcement Learning},
     journal      = {CoRR},
     volume       = {abs/2212.07489},
     year         = {2023},
   }
