Craftax
=======

JAX-based open-world survival benchmark. A significant extension of Crafter
that adds roguelike mechanics (dungeons, monsters, magic, potions) and runs
100 to 1000 times faster thanks to end-to-end JIT-compiled rollouts.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **Paradigm**
     - Single-agent
   * - **Stepping**
     - ``SINGLE_AGENT``
   * - **Backend**
     - JAX (JIT-compiled ``env.step``; scales to ``jax.lax.scan`` rollouts)
   * - **Achievements**
     - 133 (Craftax full) / 22 (Craftax-Classic, Crafter-parity)
   * - **Actions**
     - 43 discrete (full) / 17 discrete (Classic)

Installation
------------

.. code-block:: bash

   pip install -r requirements/craftax.txt

The ``requirements/craftax.txt`` recipe pulls in ``craftax``, the JAX
toolchain (``jax``, ``jaxlib``, ``flax``, ``gymnax``), and rendering helpers
(``imageio``, ``pygame``). CPU JIT works but is slow; GPU JAX is strongly
recommended for training-scale rollouts.

Registered Environments
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Environment
     - Obs mode
     - Description
   * - Craftax-Symbolic-v1
     - Symbolic
     - Full 43-action game with symbolic feature-vector observations
   * - Craftax-Pixels-v1
     - Pixels
     - Full 43-action game with 63x63x3 RGB observations
   * - Craftax-Classic-Symbolic-v1
     - Symbolic
     - Crafter-parity mechanics (17 actions) with symbolic observations
   * - Craftax-Classic-Pixels-v1
     - Pixels
     - Crafter-parity mechanics (17 actions) with 63x63x3 RGB observations

Keyboard Controls
-----------------

Craftax-Classic uses the 17-action keymap below (matches the Craftax authors'
recommended bindings, not the Crafter defaults):

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Action
     - Notes
   * - ``W A S D`` / arrows
     - Move
     - Up / Left / Down / Right
   * - ``Space``
     - Do
     - Interact with tile in front (attack, mine, collect)
   * - ``Tab``
     - Sleep
     - Regenerate health / advance day-night cycle
   * - ``R T F P``
     - Place
     - Stone, Table, Furnace, Plant
   * - ``1 2 3``
     - Craft pickaxe
     - Wood, Stone, Iron
   * - ``4 5 6``
     - Craft sword
     - Wood, Stone, Iron

Full Craftax adds these on top (extra tier + dungeons + magic):

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Key
     - Action
     - Notes
   * - ``E``
     - Rest
     - Partial sleep
   * - ``. ,``
     - Descend / Ascend
     - Move between dungeon levels
   * - ``4 8``
     - Diamond pickaxe / sword
     - Digit slot 4 is diamond-pickaxe in full Craftax (Classic uses it for wood-sword)
   * - ``Y U``
     - Iron / Diamond armour
     -
   * - ``I O``
     - Shoot / Make arrow
     -
   * - ``G H``
     - Cast Fireball / Iceball
     -
   * - ``J``
     - Place torch
     -
   * - ``Z X C V B N``
     - Drink potions
     - Red, Green, Blue, Pink, Cyan, Yellow (in that order)
   * - ``M``
     - Read book
     -
   * - ``K L``
     - Enchant sword / armour
     -
   * - ``[ ]``
     - Make torch / Level up Dexterity
     -
   * - ``- =``
     - Level up Strength / Intelligence
     -
   * - ``;``
     - Enchant bow
     -

Craftax vs Crafter
------------------

Craftax and Crafter live in **separate MOSAIC environment families**
(``EnvironmentFamily.CRAFTAX`` vs ``EnvironmentFamily.CRAFTER``) because
the runtimes differ:

* **Crafter** is NumPy-based; Craftax is JAX-based (JIT-compiled step).
* **Craftax-Classic** replicates Crafter's 22-achievement game mechanics
  exactly, so it is the drop-in fast alternative for Crafter research.
* **Full Craftax** goes further: 133 achievements, 9-level dungeon, elite
  mobs, boss fights, magic system, potions, and level-up progression.

Both can be installed side-by-side in the same virtual environment; the
family separation prevents accidental cross-registration collisions.

Citation
--------

.. code-block:: bibtex

   @inproceedings{matthews2024craftax,
     author    = {Michael Matthews and Michael Beukman and Benjamin Ellis
                  and Mikayel Samvelyan and Matthew Jackson and Samuel Coward
                  and Jakob N. Foerster},
     title     = {Craftax: A Lightning-Fast Benchmark for Open-Ended
                  Reinforcement Learning},
     booktitle = {International Conference on Machine Learning (ICML)},
     year      = {2024},
     eprint    = {2402.16801},
     archivePrefix = {arXiv},
   }
