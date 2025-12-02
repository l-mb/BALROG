import json
import logging
import re
from pathlib import Path
from typing import Any

from balrog.agents.base import BaseAgent
from balrog.client import LLMResponse

from .memory import FileMemoryBackend, MemoryEntry, TransientMemory

logger = logging.getLogger(__name__)


class BRAIDAgent(BaseAgent):
    """BALROG Recurrent Adventuring Iterative Dungeoneer.

    An agent that learns from experience through:
    - Transient memory: per-episode structured + free-form learnings
    - Persistent memory: cross-episode knowledge (file-backed, swappable to postgres/FAISS)
    - Adaptive thinking: LLM self-selects depth of reasoning per step
    """

    def __init__(self, client_factory: Any, prompt_builder: Any, config: Any):
        super().__init__(client_factory, prompt_builder)
        self.config = config

        braid_cfg = config.agent.braid
        self.transient = TransientMemory()
        self.persistent = FileMemoryBackend(Path(braid_cfg.persistent_memory_path))
        self.max_persistent_context = braid_cfg.get("max_persistent_context", 20)
        self.episode_number = braid_cfg.get("episode_number", 0)

    def act(self, obs: dict[str, Any], prev_action: str | None = None) -> LLMResponse:
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        self.prompt_builder.update_observation(obs)
        messages = self.prompt_builder.get_prompt()

        # Inject memory context and adaptive thinking instructions
        if messages:
            messages[-1].content = self._build_enhanced_prompt(messages[-1].content)

        response = self.client.generate(messages)
        return self._parse_response(response)

    def _build_enhanced_prompt(self, current_content: str) -> str:
        sections = []

        # Add persistent memory context
        persistent_context = self._format_persistent_memory()
        if persistent_context:
            sections.append(f"KNOWLEDGE FROM PREVIOUS RUNS:\n{persistent_context}")

        # Add transient memory context
        transient_context = self.transient.to_prompt_section()
        if transient_context:
            sections.append(f"EPISODE MEMORY:\n{transient_context}")

        # Current observation
        sections.append(current_content)

        # Adaptive thinking + learning management instructions
        sections.append(self._get_action_instructions())

        return "\n\n".join(sections)

    def _format_persistent_memory(self) -> str:
        entries = self.persistent.retrieve(limit=self.max_persistent_context)
        if not entries:
            return ""

        by_category: dict[str, list[str]] = {}
        for e in entries:
            by_category.setdefault(e.category, []).append(e.content)

        parts = []
        for cat, contents in by_category.items():
            parts.append(f"[{cat.upper()}]")
            parts.extend(f"- {c}" for c in contents[:5])
        return "\n".join(parts)

    def _get_action_instructions(self) -> str:
        return """
RESPONSE FORMAT:
First, assess: do you need deep analysis or is this routine?
THINKING_NEEDED: yes/no

If yes, provide analysis in <think>...</think> tags.

Then, manage memories (use KEEP if no changes needed):
<memory_updates>
TRANSIENT:
  plan: [new plan text | KEEP | CLEAR]
  plan_progress: [+add item | KEEP]
  inventory_notes: [{"item": "note"} | KEEP]
  insights: [+new insight | -remove insight | KEEP]

PERSISTENT:
  add: [{"category": "strategy|monster|item|pitfall", "content": "learning"}]
  remove: ["content substring to remove"]
</memory_updates>

Finally, your action:
<|ACTION|>YOUR_ACTION<|END|>
""".strip()

    def _parse_response(self, response: LLMResponse) -> LLMResponse:
        completion = response.completion

        # Extract thinking if present
        think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else None

        # Process memory updates
        mem_match = re.search(r"<memory_updates>(.*?)</memory_updates>", completion, re.DOTALL)
        if mem_match:
            self._process_memory_updates(mem_match.group(1))

        # Extract action
        action_match = re.search(r"<\|ACTION\|>(.*?)<\|END\|>", completion, re.DOTALL)
        action = action_match.group(1).strip() if action_match else self._fallback_action(completion)

        # Update prompt builder with reasoning for history
        if reasoning:
            self.prompt_builder.update_reasoning(reasoning)

        return response._replace(reasoning=reasoning or completion, completion=action)

    def _fallback_action(self, completion: str) -> str:
        """Extract action when tags missing. Default to 'esc' if unparseable."""
        # Try to find any action-like word at end
        lines = completion.strip().split("\n")
        for line in reversed(lines):
            line = line.strip().lower()
            if line and not line.startswith(("<", "thinking", "memory", "transient", "persistent")):
                # Return first word that looks like an action
                words = line.split()
                if words:
                    return words[-1]
        return "esc"

    def _process_memory_updates(self, mem_text: str) -> None:
        """Parse and apply memory updates from LLM response."""
        # Parse plan
        plan_match = re.search(r"plan:\s*(.+?)(?=\n\s*\w+:|$)", mem_text, re.IGNORECASE | re.DOTALL)
        if plan_match:
            val = plan_match.group(1).strip()
            if val.upper() == "CLEAR":
                self.transient.current_plan = ""
            elif val.upper() != "KEEP":
                self.transient.current_plan = val

        # Parse plan_progress additions
        progress_match = re.search(r"plan_progress:\s*(.+?)(?=\n\s*\w+:|$)", mem_text, re.IGNORECASE | re.DOTALL)
        if progress_match:
            val = progress_match.group(1).strip()
            if val.upper() != "KEEP" and val.startswith("+"):
                self.transient.plan_progress.append(val[1:].strip())

        # Parse inventory_notes
        inv_match = re.search(r"inventory_notes:\s*(.+?)(?=\n\s*\w+:|$)", mem_text, re.IGNORECASE | re.DOTALL)
        if inv_match:
            val = inv_match.group(1).strip()
            if val.upper() != "KEEP":
                try:
                    notes = json.loads(val)
                    if isinstance(notes, dict):
                        self.transient.inventory_notes.update(notes)
                except json.JSONDecodeError:
                    pass

        # Parse insights
        insights_match = re.search(r"insights:\s*(.+?)(?=\n\s*(?:PERSISTENT|$))", mem_text, re.IGNORECASE | re.DOTALL)
        if insights_match:
            val = insights_match.group(1).strip()
            if val.upper() != "KEEP":
                for line in val.split("\n"):
                    line = line.strip()
                    if line.startswith("+"):
                        self.transient.insights.append(line[1:].strip())
                    elif line.startswith("-"):
                        try:
                            self.transient.insights.remove(line[1:].strip())
                        except ValueError:
                            pass

        # Parse PERSISTENT additions
        add_match = re.search(r"add:\s*\[(.+?)\]", mem_text, re.DOTALL)
        if add_match:
            try:
                additions = json.loads(f"[{add_match.group(1)}]")
                for item in additions:
                    if isinstance(item, dict) and "category" in item and "content" in item:
                        self.persistent.store(
                            MemoryEntry(
                                category=item["category"],
                                content=item["content"],
                                source_episode=self.episode_number,
                            )
                        )
            except json.JSONDecodeError:
                logger.debug("Failed to parse persistent memory additions")

        # Parse PERSISTENT removals
        remove_match = re.search(r"remove:\s*\[(.+?)\]", mem_text, re.DOTALL)
        if remove_match:
            try:
                removals = json.loads(f"[{remove_match.group(1)}]")
                for content_substr in removals:
                    if isinstance(content_substr, str):
                        # Find and remove entries containing this substring
                        matches = self.persistent.search(content_substr, limit=1)
                        for match in matches:
                            # Find entry ID by content match
                            for eid, data in self.persistent._data.items():
                                if data["content"] == match.content:
                                    self.persistent.remove(eid)
                                    break
            except json.JSONDecodeError:
                logger.debug("Failed to parse persistent memory removals")

    def reset(self) -> None:
        """Reset for new episode - clears transient, keeps persistent."""
        super().reset()
        self.transient.reset()
        self.episode_number += 1
