SocialJax
=========

JAX-accelerated suite of sequential social dilemma environments for multi-agent
reinforcement learning.  Environments are derived from
Melting Pot 2.0 and feature mixed incentives; agents face tension between
individual self-interest and collective welfare.

.. list-table::
   :widths: 20 80
   :header-rows: 0

   * - **Paradigm**
     - Multi-agent (simultaneous)
   * - **Stepping**
     - ``SIMULTANEOUS``
   * - **Note**
     - Pure JAX implementation; requires JAX with compatible hardware backend (CPU/GPU/TPU)
   * - **Algorithms**
     - IPPO, MAPPO, MAT (see ``3rd_party/workers/jaxmarl_worker/algorithms/``)

Installation
------------

.. code-block:: bash

   pip install -e 3rd_party/environments/SocialJax/ && pip install -e ".[socialjax]"

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Environment ID
     - Agents
     - Description
   * - ``coin_game``
     - 2
     - Classic 2-player coin collection dilemma; individual vs. cooperative collection incentives
   * - ``harvest_common_open``
     - 2-6
     - Commons Harvest (open): agents harvest regrowable resources; cooperation prevents depletion
   * - ``clean_up``
     - 2-5
     - Clean Up: agents must clean pollution to unlock apple regrowth; altruism vs. free-riding
   * - ``coop_mining``
     - 2-6
     - Cooperative Mining: iron vs. gold extraction; gold requires coordination of multiple agents
   * - ``territory_open``
     - 2
     - Territory: agents claim and defend spatial zones; conflict arises at contested boundaries
   * - ``pd_arena``
     - 2
     - Prisoner's Dilemma Arena: iterated social dilemma in a grid-world setting
   * - ``mushrooms``
     - 2-4
     - Mushrooms: foraging with externalities; collecting toxic mushrooms harms neighbours
   * - ``gift``
     - 2-4
     - Gift Exchange: agents can give resources to others; tests reciprocity and trust
   * - ``lb_foraging``
     - 2-4
     - Level-Based Foraging: items require coordinated simultaneous pickup by strong-enough teams

Reward Modes
------------

All environments support two reward modes controlled at instantiation time:

- **Individual rewards** (``shared_rewards=False``): each agent receives its own signal,
  inherently encouraging selfish behaviour.
- **Common rewards** (``shared_rewards=True``): all agents share a single unified signal,
  aligning incentives and promoting cooperation.

Usage
-----

SocialJax uses its own ``make()`` factory rather than Gymnasium's ``gym.make()``:

.. code-block:: python

   import sys
   sys.path.insert(0, "3rd_party/environments/SocialJax")

   import jax
   import socialjax

   env = socialjax.make("coop_mining")          # 4 agents by default
   key = jax.random.PRNGKey(0)
   obs, state = env.reset(key)
   print(obs.keys())   # dict_keys(['agent_0', ..., 'agent_3'])

   key, subkey = jax.random.split(key)
   actions = {a: env.action_space(a).sample(subkey) for a in env.agents}
   obs, state, rewards, dones, info = env.step(subkey, state, actions)

The ``jaxmarl_worker`` integration wraps ``coop_mining`` for use with IPPO / MAPPO / MAT
training loops.  See ``3rd_party/workers/jaxmarl_worker/jaxmarl_worker/environments/socialjax/``
for the wrapper and ``3rd_party/workers/jaxmarl_worker/jaxmarl_worker/algorithms/`` for
ready-to-run training scripts.

Citation
--------

.. code-block:: bibtex

   @inproceedings{guo2026socialjax,
     author       = {Guo, Zihao and others},
     title        = {SocialJax: A Suite of Sequential Social Dilemma Environments for
                     Multi-Agent Reinforcement Learning in JAX},
     booktitle    = {International Conference on Learning Representations (ICLR)},
     year         = {2026},
     url          = {https://arxiv.org/abs/2503.14576},
   }
