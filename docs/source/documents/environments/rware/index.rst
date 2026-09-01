RWARE (Robotic Warehouse)
=========================

Cooperative multi-agent shelf delivery in a simulated warehouse.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **Paradigm**
     - Multi-agent (simultaneous)
   * - **Stepping**
     - ``SIMULTANEOUS``

Installation
------------

.. code-block:: bash

   pip install -e 3rd_party/environments/robotic-warehouse/

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Environment
     - Description
   * - rware-tiny-2ag/4ag-v2
     - Tiny warehouse (2 or 4 agents)
   * - rware-small-2ag/4ag-v2
     - Small warehouse
   * - rware-medium-2ag/4ag-v2
     - Medium warehouse (also easy/hard reward variants)
   * - rware-large-4ag/8ag-v2
     - Large warehouse (also hard reward variants)

.. figure:: /images/envs/rware/rware_size_variations.png
   :width: 100%
   :alt: RWARE size variations

   Three size variations of the multi-robot warehouse environment:
   (a) tiny / two agents, (b) small / two agents, (c) medium / four agents.
   Shelves (purple), requested shelves (teal), goal locations (dark), agents
   (orange), and an agent carrying a shelf (red).

Citation
--------

.. code-block:: bibtex

   @inproceedings{papoudakis2021rware,
     author       = {Georgios Papoudakis and Filippos Christianos and Lukas Sch{\"a}fer and Stefano V. Albrecht},
     title        = {Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks},
     booktitle    = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS)},
     year         = {2021},
   }
