Adapters API
============

Adapter classes for different environment types. Adapters bridge the
gap between raw environment APIs and MOSAIC's unified operator
interface, translating observations and actions so that any agent type
(LLM, RL, Human, or Random) can interact with any supported
environment.

The adapter layer is organised into three concerns:

- **Base Interfaces** define the lifecycle contract every adapter honours.
- **Multi-agent Adapters** wrap PettingZoo-style environments where more
  than one agent shares the same episode.
- **Paradigm Bridges** normalise stepping behaviour across single-agent,
  sequential (AEC), and simultaneous (POSG) environments.

Base Interfaces
---------------

The :class:`~gym_gui.core.adapters.base.EnvironmentAdapter` abstract class
defines the lifecycle every concrete adapter implements: ``load``,
``reset``, ``step``, ``render``, and ``close``. Concrete subclasses set
``id``, ``supported_control_modes``, and ``default_render_mode`` as class
attributes.

.. code-block:: python

   from gym_gui.core.adapters.base import (
       AdapterContext,
       EnvironmentAdapter,
   )
   from gym_gui.core.enums import ControlMode

   # Instantiate a concrete adapter subclass (see the environment-specific
   # adapters under gym_gui.core.adapters.*) and drive the lifecycle:
   context = AdapterContext(settings=None, control_mode=ControlMode.AGENT_ONLY)
   adapter: EnvironmentAdapter = MyConcreteAdapter(context)
   adapter.load()
   step = adapter.reset(seed=0)
   step = adapter.step(action=0)
   adapter.close()

EnvironmentAdapter
~~~~~~~~~~~~~~~~~~

.. autoclass:: gym_gui.core.adapters.base.EnvironmentAdapter
   :members:
   :no-index:

Multi-agent Adapters
--------------------

:class:`~gym_gui.core.adapters.pettingzoo.PettingZooAdapter` provides a
single adapter that unifies the PettingZoo AEC (turn-based) and Parallel
(simultaneous) APIs behind the standard ``load``/``reset``/``step``
lifecycle.

.. code-block:: python

   from gym_gui.core.adapters.pettingzoo import (
       PettingZooAdapter,
       PettingZooConfig,
   )
   from gym_gui.core.pettingzoo_enums import PettingZooEnvId

   config = PettingZooConfig(env_id=PettingZooEnvId.CHESS)
   adapter = PettingZooAdapter(config=config)
   adapter.load()
   step = adapter.reset(seed=0)

   # AEC games step one agent at a time via adapter.current_agent.
   action = adapter.sample_action(adapter.current_agent)
   step = adapter.step(action)

PettingZooAdapter
~~~~~~~~~~~~~~~~~

.. autoclass:: gym_gui.core.adapters.pettingzoo.PettingZooAdapter
   :members:
   :no-index:

Paradigm Bridges
----------------

:class:`~gym_gui.core.adapters.paradigm.ParadigmAdapter` abstracts the
differences between single-agent, sequential, and simultaneous stepping
so the orchestrator can drive any environment with the same loop. Use
the ``create_paradigm_adapter`` factory to auto-detect the paradigm from
the wrapped environment.

.. code-block:: python

   import gymnasium as gym
   from gym_gui.core.adapters.paradigm import create_paradigm_adapter

   env = gym.make("CartPole-v1")
   paradigm_adapter = create_paradigm_adapter(env)
   result = paradigm_adapter.reset(seed=0)
   while not paradigm_adapter.is_done():
       agents = paradigm_adapter.get_agents_to_act()
       actions = {a: env.action_space.sample() for a in agents}
       result = paradigm_adapter.step(actions)
   paradigm_adapter.close()

ParadigmAdapter
~~~~~~~~~~~~~~~

.. autoclass:: gym_gui.core.adapters.paradigm.ParadigmAdapter
   :members:
