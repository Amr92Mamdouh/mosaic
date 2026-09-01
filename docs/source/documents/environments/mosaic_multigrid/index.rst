MOSAIC MultiGrid
================

Competitive team-based multi-agent grid-world games.  Developed as part of
MOSAIC with ``view_size=3`` (agent-centric partial observability).

**New in v6.9.0:** Correct goal geometry is now enforced across all Soccer
environments (3-cell goal arc at y=4,5,6), a new ``MosaicMultiGrid-S-1v1-TeamObs-v1``
environment completes the Soccer matrix to 16 registered IDs, and JAX training
environments now require an explicit ``goal_rows`` parameter to prevent the
invisible-goals regression.

American Football provides brown field rendering, end-zone scoring, ball
stealing, and touchdown detection.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **Paradigm**
     - Multi-agent (simultaneous)
   * - **Stepping**
     - ``SIMULTANEOUS``
   * - **PyPI**
     - `mosaic-multigrid v6.9.0 <https://pypi.org/project/mosaic-multigrid/>`_

Installation
------------

.. code-block:: bash

   pip install -e ".[mosaic_multigrid]"

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Sport (grid)
     - Registered variants
   * - **Soccer** (16×11)
     - Solo (``S-G-1v0``, ``S-B-0v1``); competitive ``S-1v1``, ``S-2v2``, ``S-3v3``;
       one-sided cooperative ``S-G-2v0``/``S-G-3v0``/``S-B-0v2``/``S-B-0v3``
   * - **Basketball** (19×11)
     - Solo (``BB-G-1v0``, ``BB-B-0v1``); competitive ``BB-1v1``, ``BB-2v2``, ``BB-3v3``;
       one-sided cooperative ``BB-G-2v0``/``BB-G-3v0``/``BB-B-0v2``/``BB-B-0v3``
   * - **American Football** (16×11)
     - Solo (``AF-G-1v0``, ``AF-B-0v1``); competitive ``AF-1v1``, ``AF-2v2``, ``AF-3v3``;
       one-sided cooperative ``AF-G-2v0``/``AF-G-3v0``/``AF-B-0v2``/``AF-B-0v3``
   * - **Collect** (10×10)
     - ``C-IndAgObs`` (3 agents), ``C-1v1``, ``C-2v2``
   * - *Observation variants*
     - Every multi-agent ID ships as ``-IndAgObs`` (egocentric only) and ``-TeamObs``
       (SMAC-style teammate awareness); solo IDs are single-agent


Environment Gallery
-------------------

Each panel shows the registered variants of a sport, rendered with their
respective field/court layouts.

.. figure:: /images/envs/mosaic_multigrid/envs_S.png
   :width: 100%
   :alt: Soccer environment variants

   Soccer (16×11): solo, 1v1, 2v2, 3v3, and one-sided cooperative variants.

.. figure:: /images/envs/mosaic_multigrid/envs_BB.png
   :width: 100%
   :alt: Basketball environment variants

   Basketball (19×11): solo, 1v1, 2v2, 3v3, and one-sided cooperative variants.

.. figure:: /images/envs/mosaic_multigrid/envs_AF.png
   :width: 100%
   :alt: American Football environment variants

   American Football (16×11): solo, 1v1, 2v2, 3v3, and one-sided cooperative variants.


Environment IDs
---------------

American Football
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import gymnasium as gym
   import mosaic_multigrid.envs

   # Solo training (curriculum pre-training)
   env = gym.make('MosaicMultiGrid-AF-G-1v0-v1')
   env = gym.make('MosaicMultiGrid-AF-B-0v1-v1')

   # 1v1 competitive
   env = gym.make('MosaicMultiGrid-AF-1v1-IndAgObs-v1')

   # 2v2 competitive
   env = gym.make('MosaicMultiGrid-AF-2v2-IndAgObs-v1')
   env = gym.make('MosaicMultiGrid-AF-2v2-TeamObs-v1')  # With teammate awareness

   # 3v3 competitive
   env = gym.make('MosaicMultiGrid-AF-3v3-IndAgObs-v1')
   env = gym.make('MosaicMultiGrid-AF-3v3-TeamObs-v1')  # With teammate awareness

   # One-sided cooperative (curriculum: N teammates, no opponent)
   env = gym.make('MosaicMultiGrid-AF-G-2v0-IndAgObs-v1')
   env = gym.make('MosaicMultiGrid-AF-B-0v2-IndAgObs-v1')

**Features:**
- 16×11 brown field with alternating stripes
- White boundary lines and yard lines
- Colored end zones (green and blue)
- Touchdown scoring: walk into opponent's end zone while carrying ball
- Ball stealing: use pickup action on opponent carrying ball
- Agents cannot score in their own end zones (verified by tests)
- Custom renderer with optional HUD (agent labels, FOV highlights)

Soccer, Basketball, Collect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Soccer (16 registered IDs as of v6.9.0)
   env = gym.make('MosaicMultiGrid-S-1v1-IndAgObs-v1')
   env = gym.make('MosaicMultiGrid-S-2v2-IndAgObs-v1')
   env = gym.make('MosaicMultiGrid-S-2v2-TeamObs-v1')
   env = gym.make('MosaicMultiGrid-S-1v1-TeamObs-v1')  # new in v6.9.0

   # Basketball
   env = gym.make('MosaicMultiGrid-BB-3v3-IndAgObs-v1')
   env = gym.make('MosaicMultiGrid-BB-3v3-TeamObs-v1')

   # Collect
   env = gym.make('MosaicMultiGrid-C-IndAgObs-v1')
   env = gym.make('MosaicMultiGrid-C-2v2-IndAgObs-v1')
   env = gym.make('MosaicMultiGrid-C-1v1-IndAgObs-v1')
   env = gym.make('MosaicMultiGrid-C-2v2-TeamObs-v1')


Action Space
------------

All environments use the same action space (Discrete(8)):

.. list-table::
   :header-rows: 1
   :widths: 10 20 70

   * - ID
     - Action
     - Description
   * - 0
     - NOOP
     - No operation (AEC compatibility)
   * - 1
     - LEFT
     - Rotate 90° counter-clockwise
   * - 2
     - RIGHT
     - Rotate 90° clockwise
   * - 3
     - FORWARD
     - Move one cell in facing direction
   * - 4
     - PICKUP
     - Pick up ball / steal from opponent
   * - 5
     - DROP
     - Drop ball / pass to teammate / shoot
   * - 6
     - TOGGLE
     - Toggle/activate object
   * - 7
     - DONE
     - Signal task completion


Gameplay Rules
--------------

American Football
~~~~~~~~~~~~~~~~~

**Objective:** Score touchdowns by carrying the ball into the opponent's end zone.

**Field Layout:**
- 16×11 brown field with white yard lines
- Green end zone (column 1): Defended by Team 0, scored on by Team 1
- Blue end zone (column 14): Defended by Team 1, scored on by Team 0

**How to Play:**
1. **Pick up the ball:** Use PICKUP (action 4) when facing the ball
2. **Carry the ball:** Move with FORWARD (action 3) while carrying
3. **Score touchdown:** Walk into the opponent's end zone while carrying the ball
4. **Ball stealing:** Use PICKUP (action 4) when facing an opponent carrying the ball
5. **Pass to teammate:** Use DROP (action 5) when facing a teammate (ball teleports to them)
6. **Important:** You CANNOT score in your own end zone

**Scoring:**
- Touchdown = +1 point for scoring team
- Zero-sum: Opponent receives -1 point
- Episode terminates after touchdown
- Ball respawns in midfield after touchdown

Soccer
~~~~~~

**Objective:** Score goals by shooting the ball into the opponent's goal.

**Field Layout:**
- 16×11 FIFA-style green field
- Goals at left and right ends of the field
- First team to 2 goals wins

**How to Play:**
1. **Pick up the ball:** Use PICKUP (action 4) when facing the ball
2. **Dribble:** Move with FORWARD (action 3) while carrying
3. **Shoot:** Use DROP (action 5) when facing the opponent's goal
4. **Pass:** Use DROP (action 5) when facing a teammate
5. **Steal:** Use PICKUP (action 4) when facing an opponent with the ball

**Scoring:**
- Goal = +1 point for scoring team
- Ball respawns at center after each goal
- Episode continues until one team reaches 2 goals

Basketball
~~~~~~~~~~

**Objective:** Score baskets by shooting the ball into the opponent's hoop.

**Field Layout:**
- 19×11 court with basketball markings
- Hoops at left and right ends of the court

**How to Play:**
1. **Pick up the ball:** Use PICKUP (action 4) when facing the ball
2. **Dribble:** Move with FORWARD (action 3) while carrying
3. **Shoot:** Use DROP (action 5) when facing the opponent's hoop
4. **Pass:** Use DROP (action 5) when facing a teammate
5. **Steal:** Use PICKUP (action 4) when facing an opponent with the ball

**Scoring:**
- Basket = +1 point for scoring team
- Ball respawns at center after each basket

Collect
~~~~~~~

**Objective:** Collect more balls than opponents before time runs out.

**Field Layout:**
- Multiple colored balls scattered on the field
- Each ball has a designated collection zone

**How to Play:**
1. **Pick up a ball:** Use PICKUP (action 4) when facing a ball
2. **Carry to zone:** Move with FORWARD (action 3) to the ball's collection zone
3. **Score:** Use DROP (action 5) in the correct collection zone
4. **Steal:** Use PICKUP (action 4) when facing an opponent with a ball

**Scoring:**
- Each ball collected = +1 point
- Episode ends when all balls are collected or time limit reached
- Agent/team with most points wins


Citation
--------


.. code-block:: bibtex 
  
    @article{mousa2026mosaic,
      title = {MOSAIC MultiGrid: Research-Grade Multi-Agent Gridworld Environments},
      author = {Mousa, Abdulhamid},
      journal = {GitHub repository},
      year = {2026},
      url = {https://github.com/Abdulhamid97Mousa/mosaic_multigrid},
    }

    @misc{gym_multigrid,
      author = {Fickinger, Arnaud},
      title = {Multi-Agent Gridworld Environment for OpenAI Gym},
      year = {2020},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/ArnaudFickinger/gym-multigrid}},
    }

    @article{oguntola2023theory,
      title = {Theory of Mind as Intrinsic Motivation for Multi-Agent Reinforcement Learning},
      author = {Oguntola, Ini and Campbell, Joseph and Stepputtis, Simon and Sycara, Katia},
      journal = {arXiv preprint arXiv:2307.01158},
      year = {2023},
      url = {https://github.com/ini/multigrid},
    }

    @misc{mosaic_multigrid,
      author = {Mousa, Abdulhamid},
      title = {mosaic\_multigrid: Research-Grade Multi-Agent Gridworld Environments},
      year = {2026},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/Abdulhamid97Mousa/mosaic_multigrid}},
    }

    @article{terry2021pettingzoo,
      title = {PettingZoo: Gym for Multi-Agent Reinforcement Learning},
      author = {Terry, J. K and Black, Benjamin and Grammel, Nathaniel and Jayakumar, Mario
                and Hari, Ananth and Sullivan, Ryan and Santos, Luis S and Dieffendahl, Clemens
                and Horsch, Caroline and Perez-Vicente, Rodrigo and Williams, Niall L
                and Lokesh, Yashas and Ravi, Praveen},
      journal = {Advances in Neural Information Processing Systems},
      volume = {34},
      pages = {2242--2254},
      year = {2021},
      url = {https://pettingzoo.farama.org/},
    }