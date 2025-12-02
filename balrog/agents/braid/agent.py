import json
import logging
import re
from pathlib import Path
from typing import Any

from balrog.agents.base import BaseAgent
from balrog.client import LLMResponse

from .memory import FileMemoryBackend, MemoryEntry, MemoryScope

logger = logging.getLogger(__name__)


class BRAIDAgent(BaseAgent):
    """BALROG Recurrent Adventuring Iterative Dungeoneer.

    An agent that learns from experience through unified memory:
    - Episode memory: per-episode insights (scope=episode)
    - Persistent memory: cross-episode knowledge (scope=persistent)
    - Label filtering: LLM can enable/disable labels to control what's shown
    - Adaptive thinking: LLM self-selects depth of reasoning per step
    """

    _SYSTEM_PROMPT_SUFFIX = """
RESPONSE FORMAT:
1. Assess: THINKING_NEEDED: yes/no
2. If yes: <think>your analysis</think>
3. Memory updates:
<memory_updates>
add: [{"scope": "episode|persistent", "tags": "t1,t2", "prio": 5, "content": "..."}]
remove: ["entry_id"]
enable_labels: ["tag"] | disable_labels: ["tag"] | reset_labels: true
</memory_updates>
4. Action: <|ACTION|>your_action<|END|>

MEMORY:
- scope: episode (cleared each ep) | persistent (survives)
- prio: 1-9, higher shown first when limit reached (default 5)
- enable/disable_labels: filter what's shown; reset_labels: true to show all

EFFICIENCY: content/tags for agent only - abbreviate freely.
""".strip()

    def __init__(self, client_factory: Any, prompt_builder: Any, config: Any):
        super().__init__(client_factory, prompt_builder)
        self.config = config

        num_workers = config.eval.get("num_workers", 1)
        if num_workers > 1:
            raise ValueError(
                f"BRAIDAgent with FileMemoryBackend requires eval.num_workers=1 (got {num_workers}). "
                "Parallel workers would corrupt shared JSON storage and episode numbering."
            )

        braid_cfg = config.agent.braid
        self.memory = FileMemoryBackend(Path(braid_cfg.persistent_memory_path))
        self.max_memory_context = braid_cfg.get("max_persistent_context", 40)
        self.episode_number = self.memory.max_episode()
        self._enabled_labels: set[str] | None = None  # None = all labels enabled

    def build_system_prompt(self, env_instruction: str) -> str:
        """Append BRAID response format instructions to environment prompt."""
        return f"{env_instruction}\n\n{self._SYSTEM_PROMPT_SUFFIX}"

    def act(self, obs: dict[str, Any], prev_action: str | None = None) -> LLMResponse:
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)
        messages = self.prompt_builder.get_prompt()

        if messages:
            messages[-1].content = self._build_enhanced_prompt(messages[-1].content)

        response = self.client.generate(messages)
        return self._parse_response(response)

    def _get_label_stats(
        self, scope: MemoryScope, episode: int | None = None
    ) -> dict[str, Any]:
        """Get label statistics for prompt building."""
        counts = self.memory.count_by_tag(scope=scope, episode=episode)
        all_labels = set(counts.keys())

        if self._enabled_labels is None:
            enabled = all_labels
            disabled: set[str] = set()
        else:
            enabled = self._enabled_labels & all_labels
            disabled = all_labels - self._enabled_labels

        return {
            "enabled": {lbl: counts[lbl] for lbl in enabled},
            "disabled": {lbl: counts[lbl] for lbl in disabled},
            "total_enabled": sum(counts[lbl] for lbl in enabled),
            "total_disabled": sum(counts[lbl] for lbl in disabled),
        }

    def _format_label_controls(
        self, persistent_stats: dict[str, Any], episode_stats: dict[str, Any]
    ) -> str:
        """Format the label filter controls for the prompt."""
        lines = ["MEMORY LABEL FILTERS:"]

        # Merge counts from both scopes
        all_enabled: dict[str, int] = {}
        all_disabled: dict[str, int] = {}

        for lbl, cnt in persistent_stats["enabled"].items():
            all_enabled[lbl] = all_enabled.get(lbl, 0) + cnt
        for lbl, cnt in episode_stats["enabled"].items():
            all_enabled[lbl] = all_enabled.get(lbl, 0) + cnt
        for lbl, cnt in persistent_stats["disabled"].items():
            all_disabled[lbl] = all_disabled.get(lbl, 0) + cnt
        for lbl, cnt in episode_stats["disabled"].items():
            all_disabled[lbl] = all_disabled.get(lbl, 0) + cnt

        if all_enabled:
            enabled_str = ", ".join(f"{lbl}({cnt})" for lbl, cnt in sorted(all_enabled.items()))
            lines.append(f"Enabled: {enabled_str}")
        else:
            lines.append("Enabled: (none)")

        if all_disabled:
            disabled_str = ", ".join(f"{lbl}({cnt})" for lbl, cnt in sorted(all_disabled.items()))
            lines.append(f"Disabled (hidden): {disabled_str}")

        total_hidden = persistent_stats["total_disabled"] + episode_stats["total_disabled"]
        if total_hidden > 0:
            lines.append(f"({total_hidden} entries hidden by disabled labels)")

        return "\n".join(lines)

    def _build_enhanced_prompt(self, current_content: str) -> str:
        """Build prompt with sections ordered for optimal cache hits.

        Order: persistent (stable) → episode (per-episode) → labels (dynamic) → observation
        """
        sections = []

        # Persistent memory first (most stable across steps - best cache prefix)
        persistent, p_shown, p_more = self._format_memory(MemoryScope.PERSISTENT)
        if persistent:
            header = f"PERSISTENT ({p_shown}"
            if p_more > 0:
                header += f"+{p_more} hidden due to limits"
            header += "):"
            sections.append(f"{header}\n{persistent}")

        # Episode memory (stable within episode)
        episode, e_shown, e_more = self._format_memory(MemoryScope.EPISODE, episode=self.episode_number)
        if episode:
            header = f"EPISODE ({e_shown}"
            if e_more > 0:
                header += f"+{e_more} hidden due to limits"
            header += "):"
            sections.append(f"{header}\n{episode}")

        # Label controls (can change within episode)
        persistent_stats = self._get_label_stats(MemoryScope.PERSISTENT)
        episode_stats = self._get_label_stats(MemoryScope.EPISODE, episode=self.episode_number)
        sections.append(self._format_label_controls(persistent_stats, episode_stats))

        # Current observation (changes every step)
        sections.append(current_content)

        # Minimal action reminder (full instructions in system prompt)
        sections.append(self._get_action_instructions())

        return "\n\n".join(sections)

    def _format_memory(
        self, scope: MemoryScope, episode: int | None = None
    ) -> tuple[str, int, int]:
        """Format memory entries, return (formatted_str, shown_count, more_available)."""
        filter_tags = self._enabled_labels  # None means all

        entries = self.memory.retrieve(
            tags=filter_tags, scope=scope, episode=episode, limit=self.max_memory_context
        )
        total = self.memory.count(tags=filter_tags, scope=scope, episode=episode)
        more_available = total - len(entries)

        if not entries:
            return "", 0, more_available

        parts = []
        for e in entries:
            parts.append(f"[{e.entry_id}:p{e.priority}] ({e.tags}) {e.content}")
        return "\n".join(parts), len(entries), more_available

    def _get_action_instructions(self) -> str:
        """Minimal reminder - full instructions in system prompt."""
        if self._enabled_labels is None:
            filter_state = "labels: all"
        elif not self._enabled_labels:
            filter_state = "labels: none (all hidden)"
        else:
            filter_state = f"labels: {','.join(sorted(self._enabled_labels))}"
        return f"[{filter_state}] <|ACTION|>action<|END|>"

    def _parse_response(self, response: LLMResponse) -> LLMResponse:
        completion = response.completion

        think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else None

        mem_match = re.search(r"<memory_updates>(.*?)</memory_updates>", completion, re.DOTALL)
        if mem_match:
            self._process_memory_updates(mem_match.group(1))

        action_match = re.search(r"<\|ACTION\|>(.*?)<\|END\|>", completion, re.DOTALL)
        action = action_match.group(1).strip() if action_match else self._fallback_action(completion)

        if reasoning:
            self.prompt_builder.update_reasoning(reasoning)

        return response._replace(reasoning=reasoning or completion, completion=action)

    def _fallback_action(self, completion: str) -> str:
        """Extract action when tags missing. Default to 'esc' if unparseable."""
        lines = completion.strip().split("\n")
        for line in reversed(lines):
            line = line.strip().lower()
            if line and not line.startswith(("<", "thinking", "memory", "add", "remove", "enable", "disable", "reset")):
                words = line.split()
                if words:
                    return words[-1]
        return "esc"

    def _process_memory_updates(self, mem_text: str) -> None:
        """Parse and apply memory updates from LLM response."""
        # Parse additions
        add_match = re.search(r"add:\s*\[(.+?)\]", mem_text, re.DOTALL)
        if add_match:
            try:
                additions = json.loads(f"[{add_match.group(1)}]")
                for item in additions:
                    if not isinstance(item, dict):
                        continue
                    tags = item.get("tags", "").lower().strip()
                    content = item.get("content", "")
                    scope_str = item.get("scope", "persistent").lower()
                    prio = max(1, min(9, int(item.get("prio", 5))))

                    if not tags or not content:
                        continue

                    scope = MemoryScope.EPISODE if scope_str == "episode" else MemoryScope.PERSISTENT

                    self.memory.store(
                        MemoryEntry(
                            tags=tags,
                            content=content,
                            scope=scope,
                            priority=prio,
                            source_episode=self.episode_number,
                        )
                    )
            except json.JSONDecodeError:
                logger.debug("Failed to parse memory additions")

        # Parse removals (by ID)
        remove_match = re.search(r"remove:\s*\[(.+?)\]", mem_text, re.DOTALL)
        if remove_match:
            try:
                removals = json.loads(f"[{remove_match.group(1)}]")
                for entry_id in removals:
                    if isinstance(entry_id, str):
                        self.memory.remove(entry_id)
            except json.JSONDecodeError:
                logger.debug("Failed to parse memory removals")

        # Parse enable_labels
        enable_match = re.search(r"enable_labels:\s*\[(.+?)\]", mem_text, re.DOTALL)
        if enable_match:
            try:
                labels = json.loads(f"[{enable_match.group(1)}]")
                labels = {str(lbl).lower().strip() for lbl in labels if isinstance(lbl, str)}
                if labels:
                    if self._enabled_labels is None:
                        # Already all enabled, just stay that way
                        pass
                    else:
                        self._enabled_labels |= labels
            except json.JSONDecodeError:
                logger.debug("Failed to parse enable_labels")

        # Parse disable_labels
        disable_match = re.search(r"disable_labels:\s*\[(.+?)\]", mem_text, re.DOTALL)
        if disable_match:
            try:
                labels = json.loads(f"[{disable_match.group(1)}]")
                labels = {str(lbl).lower().strip() for lbl in labels if isinstance(lbl, str)}
                if labels:
                    if self._enabled_labels is None:
                        # First disable: enable everything except these
                        all_tags = self.memory.all_tags()
                        self._enabled_labels = all_tags - labels
                    else:
                        self._enabled_labels -= labels
            except json.JSONDecodeError:
                logger.debug("Failed to parse disable_labels")

        # Parse reset_labels
        reset_match = re.search(r"reset_labels:\s*(true|false)", mem_text, re.IGNORECASE)
        if reset_match and reset_match.group(1).lower() == "true":
            self._enabled_labels = None

    def reset(self) -> None:
        """Reset for new episode. Episode memories and label filters reset."""
        super().reset()
        self.episode_number += 1
        self._enabled_labels = None  # Reset label filter each episode
