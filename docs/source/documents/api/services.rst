Services API
============

Service classes providing MOSAIC's operator business logic. The classes
below are grouped by role:

- **Configuration** dataclasses describe *what* to run: which workers,
  which environment, which players, and which policy bindings.
- **Runtime Services** carry the configuration through the operator
  lifecycle: launching subprocess workers, driving scripted experiments,
  routing actions per agent, and coordinating actors.

Configuration
-------------

The configuration layer is built around
:class:`~gym_gui.services.operator.OperatorConfig`, which owns a
mapping of ``player_id`` to
:class:`~gym_gui.services.operator.WorkerAssignment`. Multi-agent RL
runs that share a checkpoint across agents use
:class:`~gym_gui.services.operator.LinkGroup` for policy sharing, and
per-agent policy routing is described by
:class:`~gym_gui.services.policy_mapping.AgentPolicyBinding`.

Use the factory methods on ``OperatorConfig`` rather than instantiating
the dataclass directly; the factories build the ``workers`` dict for
you and set the right defaults for single- vs. multi-agent runs.

.. code-block:: python

   from gym_gui.services.operator import (
       LinkGroup,
       OperatorConfig,
       WorkerAssignment,
   )
   from gym_gui.services.policy_mapping import AgentPolicyBinding

   # Single-agent operator (Gymnasium-style environment).
   single = OperatorConfig.single_agent(
       operator_id="op_0",
       display_name="GPT-4 Agent",
       worker_id="balrog_worker",
       worker_type="llm",
       env_name="babyai",
       task="BabyAI-GoToRedBall-v0",
       settings={"client_name": "vllm", "model_id": "meta-llama/Llama-3.1-8B-Instruct"},
   )

   # Multi-agent operator (PettingZoo chess).
   multi = OperatorConfig.multi_agent(
       operator_id="op_1",
       display_name="Chess Match",
       env_name="pettingzoo",
       task="chess_v6",
       player_workers={
           "player_0": WorkerAssignment(worker_id="balrog_worker", worker_type="llm"),
           "player_1": WorkerAssignment(worker_id="cleanrl_worker", worker_type="rl"),
       },
   )

   # Optional: share one RL policy across several agents.
   link = LinkGroup(
       group_id="op_1_link_0",
       primary_agent="player_0",
       linked_agents=["player_1"],
       policy_path="/path/to/checkpoint.pt",
       algorithm="IPPO",
   )

   # Per-agent policy routing for PolicyMappingService.
   binding = AgentPolicyBinding(
       agent_id="player_0",
       policy_id="human_keyboard",
   )

OperatorConfig
~~~~~~~~~~~~~~

.. autoclass:: gym_gui.services.operator.OperatorConfig
   :members:
   :no-index:

WorkerAssignment
~~~~~~~~~~~~~~~~

.. autoclass:: gym_gui.services.operator.WorkerAssignment
   :members:
   :no-index:

LinkGroup
~~~~~~~~~

.. autoclass:: gym_gui.services.operator.LinkGroup
   :members:
   :no-index:

AgentPolicyBinding
~~~~~~~~~~~~~~~~~~

.. autoclass:: gym_gui.services.policy_mapping.AgentPolicyBinding
   :members:
   :no-index:

Runtime Services
----------------

The runtime services turn a configuration into a live operator.
:class:`~gym_gui.services.operator_launcher.OperatorLauncher` spawns and
tracks worker subprocesses.
:class:`~gym_gui.services.operator_script_execution_manager.OperatorScriptExecutionManager`
drives scripted multi-episode experiments.
:class:`~gym_gui.services.policy_mapping.PolicyMappingService` picks the
right actor for each agent when the environment is multi-agent, and
:class:`~gym_gui.services.actor.ActorService` is the underlying registry
of controllers.

.. code-block:: python

   from gym_gui.services.actor import ActorService
   from gym_gui.services.operator import OperatorConfig, WorkerAssignment
   from gym_gui.services.operator_launcher import OperatorLauncher
   from gym_gui.services.policy_mapping import PolicyMappingService

   # Launch a subprocess worker for an operator.
   launcher = OperatorLauncher()
   config = OperatorConfig.single_agent(
       operator_id="op_0",
       display_name="Random Baseline",
       worker_id="random_worker",
       worker_type="random",
       env_name="babyai",
       task="BabyAI-GoToRedBall-v0",
   )
   handle = launcher.launch_operator(config)
   if handle.is_running:
       print(f"Operator running with PID {handle.pid}")
   launcher.stop_operator("op_0")

   # Route actions per agent for a multi-agent operator.
   actors = ActorService()
   mapping = PolicyMappingService(actors)
   mapping.set_agents(["player_0", "player_1"])

OperatorLauncher
~~~~~~~~~~~~~~~~

.. autoclass:: gym_gui.services.operator_launcher.OperatorLauncher
   :members:

OperatorScriptExecutionManager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: gym_gui.services.operator_script_execution_manager.OperatorScriptExecutionManager
   :members:

PolicyMappingService
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: gym_gui.services.policy_mapping.PolicyMappingService
   :members:

ActorService
~~~~~~~~~~~~

.. autoclass:: gym_gui.services.actor.ActorService
   :members:
