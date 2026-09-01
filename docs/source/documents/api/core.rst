Core API
========

Core enums, capability descriptors, and Qt widget classes used
throughout MOSAIC. The autoclassed entries live in
``gym_gui.core``; the Qt widget entries at the bottom of this page are
maintained by hand because Sphinx cannot introspect Qt classes in a
headless build.

Enums
-----

:class:`~gym_gui.core.enums.SteppingParadigm` names the three stepping
models MOSAIC recognises: single-agent Gymnasium, sequential (AEC)
PettingZoo, and simultaneous (POSG) PettingZoo Parallel. It is
independent of who controls the agents and of the underlying
environment library.

.. code-block:: python

   from gym_gui.core.enums import SteppingParadigm

   paradigm = SteppingParadigm.SEQUENTIAL
   assert paradigm.value == "sequential"

SteppingParadigm
~~~~~~~~~~~~~~~~

.. autoclass:: gym_gui.core.enums.SteppingParadigm
   :members:

Capability Descriptors
----------------------

:class:`~gym_gui.core.adapters.base.WorkerCapabilities` is a frozen
dataclass that declares which stepping paradigms, environment types,
and action or observation spaces a worker supports. The
WorkerOrchestrator uses it to match environments to compatible workers
before launching a run.

.. code-block:: python

   from gym_gui.core.adapters.base import WorkerCapabilities
   from gym_gui.core.enums import SteppingParadigm

   caps = WorkerCapabilities(
       stepping_paradigm=SteppingParadigm.SINGLE_AGENT,
       env_types=("gymnasium",),
       action_spaces=("discrete", "continuous"),
       max_agents=1,
   )
   assert caps.supports_paradigm(SteppingParadigm.SINGLE_AGENT)
   assert caps.supports_env_type("gymnasium")

WorkerCapabilities
~~~~~~~~~~~~~~~~~~

.. autoclass:: gym_gui.core.adapters.base.WorkerCapabilities
   :members:
   :no-index:

UI Widgets
----------

.. note::

   Qt widget classes cannot be auto-documented by Sphinx in a headless
   environment. The API signatures below are maintained manually.

PlayerAssignmentPanel
~~~~~~~~~~~~~~~~~~~~~

.. py:class:: gym_gui.ui.widgets.operator_config_widget.PlayerAssignmentPanel(env_family, env_id, num_agents, agent_ids=None, agent_labels=None, parent=None)

   Panel showing all agent/player assignments for a multi-agent environment.
   Contains one :class:`PlayerAssignmentRow` per agent slot.

   .. py:attribute:: assignments_changed
      :type: pyqtSignal

      Emitted when any player assignment row changes (type, worker, or settings).

   .. py:method:: get_worker_assignments() -> Dict[str, WorkerAssignment]

      Return a dict mapping each ``player_id`` to its :class:`~gym_gui.services.operator.WorkerAssignment`.

   .. py:method:: has_llm_agent() -> bool

      Return ``True`` if at least one agent row has Type set to **LLM**.

   .. py:method:: set_vllm_servers(servers)

      Propagate the current list of vLLM servers to every row.

PlayerAssignmentRow
~~~~~~~~~~~~~~~~~~~

.. py:class:: gym_gui.ui.widgets.operator_config_widget.PlayerAssignmentRow(player_id, player_label, parent=None)

   Single row inside a :class:`PlayerAssignmentPanel`. Exposes a
   **Type** dropdown (LLM / RL / Human / Random), a **Worker** dropdown,
   and type-specific settings (LLM provider/model, RL policy path, etc.).

   .. py:attribute:: assignment_changed
      :type: pyqtSignal

      Emitted when the user changes any field in this row.

   .. py:method:: get_assignment() -> WorkerAssignment

      Build and return a :class:`~gym_gui.services.operator.WorkerAssignment`
      from the current UI state. The ``Random`` type maps to
      ``worker_type="random"`` and ``Passive`` maps to ``worker_type="passive"``.

OperatorConfigWidget
~~~~~~~~~~~~~~~~~~~~

.. py:class:: gym_gui.ui.widgets.operator_config_widget.OperatorConfigWidget(operator_id, parent=None)

   Per-operator configuration widget. For multi-agent environments it
   creates a :class:`PlayerAssignmentPanel` and an environment-specific
   settings section (observation mode, coordination strategy for
   MultiGrid / MeltingPot).

   .. py:method:: get_config() -> OperatorConfig

      Build an :class:`~gym_gui.services.operator.OperatorConfig` from
      the current UI state (single-agent or multi-agent depending on the
      selected environment).
