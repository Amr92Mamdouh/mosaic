PyBullet Drones
===============

Quadcopter control environments with realistic physics simulation.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **Paradigm**
     - Single-agent / Multi-agent
   * - **Stepping**
     - ``SINGLE_AGENT`` / ``SIMULTANEOUS``

Installation
------------

.. code-block:: bash

   pip install -e ".[pybullet-drones]"

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Environment
     - Description
   * - hover-aviary-v0
     - Single drone hovering task
   * - multihover-aviary-v0
     - Multi-drone formation hovering
   * - ctrl-aviary-v0
     - Low-level motor control
   * - velocity-aviary-v0
     - Velocity-based control

Citation
--------

.. code-block:: bibtex

   @inproceedings{panerati2021learning,
     author       = {Jacopo Panerati and Hehui Zheng and SiQi Zhou and James Xu and Amanda Prorok and Angela P. Schoellig},
     title        = {Learning to Fly -- a Gym Environment with PyBullet Physics for Reinforcement Learning of Multi-agent Quadrotor Control},
     booktitle    = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
     year         = {2021},
   }
