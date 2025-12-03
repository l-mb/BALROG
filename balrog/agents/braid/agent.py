import json
import re
from pathlib import Path
from typing import Any

from balrog.agents.base import BaseAgent
from balrog.client import LLMResponse

from .journal import BraidJournal
from .memory import FileMemoryBackend, MemoryEntry, MemoryScope


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

EFFICIENCY: content/tags for agent only - abbreviate freely, as terse as possible, disregard human readability
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

        # Override history limit if specified in braid config
        if "max_text_history" in braid_cfg:
            self.prompt_builder.max_text_history = braid_cfg.max_text_history

        # Initialize journal for debugging/monitoring
        journal_path = braid_cfg.get("journal_path")
        log_full_prompt = braid_cfg.get("journal_full_prompt", False)
        self._journal_memory_details = braid_cfg.get("journal_memory_details", False)
        self.journal = BraidJournal(
            path=journal_path,
            enabled=journal_path is not None,
            log_full_prompt=log_full_prompt,
        )
        self.journal.set_episode(self.episode_number)

        # Track memory update counts and details for journal
        self._mem_adds = 0
        self._mem_removes = 0
        self._label_changes = False
        self._added_entries: list[dict[str, Any]] = []
        self._removed_ids: list[str] = []

    def build_system_prompt(self, env_instruction: str) -> str:
        """Append BRAID response format instructions to environment prompt."""
        return f"{env_instruction}\n\n{self._SYSTEM_PROMPT_SUFFIX}"

    def _extract_screen(self, obs: dict[str, Any]) -> str | None:
        """Extract ASCII screen from observation (NetHack tty_chars)."""
        # Try to get tty_chars from various observation structures
        tty_chars = obs.get("tty_chars")
        if tty_chars is None:
            # May be nested under "text" or similar
            text_obs = obs.get("text", {})
            if isinstance(text_obs, dict):
                tty_chars = text_obs.get("tty_chars")
        if tty_chars is None:
            return None
        # Render as ASCII string
        try:
            rows, cols = tty_chars.shape
            lines = []
            for i in range(rows):
                line = "".join(chr(tty_chars[i, j]) for j in range(cols))
                lines.append(line)
            return "\n".join(lines)
        except (AttributeError, TypeError):
            return None

    def act(self, obs: dict[str, Any], prev_action: str | None = None) -> LLMResponse:
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)
        messages = self.prompt_builder.get_prompt()

        if messages:
            messages[-1].content = self._build_enhanced_prompt(messages[-1].content)

        # Log screen if available (before request for context)
        screen = self._extract_screen(obs)
        if screen:
            self.journal.log_screen(screen)

        # Extract observation text for journal
        obs_text = obs.get("text", {}).get("short_term_context", "")
        self.journal.log_request_start(messages, observation_text=obs_text)

        response = self.client.generate(messages)
        return self._parse_response(response)

    def _build_enhanced_prompt(self, current_content: str) -> str:
        """Build prompt optimized for cache hits and minimal queries.

        Order: persistent (stable) → episode (per-episode) → labels → observation
        """
        sections = []

        # Retrieve entries for each scope
        p_entries = self.memory.retrieve(
            tags=self._enabled_labels, scope=MemoryScope.PERSISTENT, limit=self.max_memory_context
        )
        e_entries = self.memory.retrieve(
            tags=self._enabled_labels, scope=MemoryScope.EPISODE,
            episode=self.episode_number, limit=self.max_memory_context
        )

        if p_entries:
            lines = [f"[{e.entry_id}] (prio:{e.priority}) (tags: {e.tags}) {e.content}" for e in p_entries]
            header = f"PERSISTENT ({len(p_entries)}"
            # Only count hidden if we hit the limit
            if len(p_entries) >= self.max_memory_context:
                total = self.memory.count(tags=self._enabled_labels, scope=MemoryScope.PERSISTENT)
                hidden = total - len(p_entries)
                if hidden > 0:
                    header += f"+{hidden} hidden due to limit"
            header += "):"
            sections.append(f"{header}\n" + "\n".join(lines))

        if e_entries:
            lines = [f"[{e.entry_id}] (prio:{e.priority}) (tags: {e.tags}) {e.content}" for e in e_entries]
            header = f"EPISODE ({len(e_entries)}"
            if len(e_entries) >= self.max_memory_context:
                total = self.memory.count(
                    tags=self._enabled_labels, scope=MemoryScope.EPISODE, episode=self.episode_number
                )
                hidden = total - len(e_entries)
                if hidden > 0:
                    header += f"+{hidden} hidden due to limit"
            header += "):"
            sections.append(f"{header}\n" + "\n".join(lines))

        # Compute label stats from already-retrieved entries (no extra queries)
        label_info = self._compute_label_summary(p_entries + e_entries)
        if label_info:
            sections.append(label_info)

        sections.append(current_content)
        sections.append(self._get_action_instructions())

        return "\n\n".join(sections)

    def _compute_label_summary(self, entries: list[MemoryEntry]) -> str:
        """Compute label summary from already-retrieved entries (no extra queries)."""
        if not entries:
            return ""
        # Count tags from retrieved entries
        tag_counts: dict[str, int] = {}
        for e in entries:
            for tag in e.tags.split(","):
                tag = tag.strip()
                if tag:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
        if not tag_counts:
            return ""
        tags_str = " ".join(f"{t}:{c}" for t, c in sorted(tag_counts.items()))
        return f"[tags: {tags_str}]"

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

        # Reset memory update counters and details
        self._mem_adds = 0
        self._mem_removes = 0
        self._label_changes = False
        self._added_entries = []
        self._removed_ids = []

        mem_match = re.search(r"<memory_updates>(.*?)</memory_updates>", completion, re.DOTALL)
        if mem_match:
            self._process_memory_updates(mem_match.group(1))

        action_match = re.search(r"<\|ACTION\|>(.*?)<\|END\|>", completion, re.DOTALL)
        action = action_match.group(1).strip() if action_match else self._fallback_action(completion)

        if reasoning:
            self.prompt_builder.update_reasoning(reasoning)

        # Log response to journal
        self.journal.log_response(
            action=action,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            reasoning=reasoning,
        )
        self.journal.log_memory_update(
            adds=self._mem_adds,
            removes=self._mem_removes,
            label_changes=self._label_changes,
            added_entries=self._added_entries if self._journal_memory_details else None,
            removed_ids=self._removed_ids if self._journal_memory_details else None,
        )

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

                    entry_id = self.memory.store(
                        MemoryEntry(
                            tags=tags,
                            content=content,
                            scope=scope,
                            priority=prio,
                            source_episode=self.episode_number,
                        )
                    )
                    self._mem_adds += 1
                    if self._journal_memory_details:
                        self._added_entries.append({
                            "id": entry_id,
                            "scope": scope_str,
                            "tags": tags,
                            "prio": prio,
                            "content": content,
                        })
            except (json.JSONDecodeError, ValueError):
                self.journal.log_error("Failed to parse memory additions")

        # Parse removals (by ID)
        remove_match = re.search(r"remove:\s*\[(.+?)\]", mem_text, re.DOTALL)
        if remove_match:
            try:
                removals = json.loads(f"[{remove_match.group(1)}]")
                for entry_id in removals:
                    if isinstance(entry_id, str):
                        self.memory.remove(entry_id)
                        self._mem_removes += 1
                        if self._journal_memory_details:
                            self._removed_ids.append(entry_id)
            except json.JSONDecodeError:
                self.journal.log_error("Failed to parse memory removals")

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
                    self._label_changes = True
            except json.JSONDecodeError:
                self.journal.log_error("Failed to parse enable_labels")

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
                    self._label_changes = True
            except json.JSONDecodeError:
                self.journal.log_error("Failed to parse disable_labels")

        # Parse reset_labels
        reset_match = re.search(r"reset_labels:\s*(true|false)", mem_text, re.IGNORECASE)
        if reset_match and reset_match.group(1).lower() == "true":
            self._enabled_labels = None
            self._label_changes = True

    def reset(self) -> None:
        """Reset for new episode. Episode memories and label filters reset."""
        super().reset()
        self.episode_number += 1
        self._enabled_labels = None  # Reset label filter each episode
        self.journal.reset_episode(self.episode_number)
