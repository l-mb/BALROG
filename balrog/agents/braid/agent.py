import json
import re
import time
from pathlib import Path
from typing import Any

from balrog.agents.base import BaseAgent
from balrog.client import LLMResponse

from .storage import BraidStorage, MemoryEntry, MemoryScope


class BRAIDAgent(BaseAgent):
    """BALROG Recurrent Adventuring Iterative Dungeoneer.

    An agent that learns from experience through unified memory:
    - Episode memory: per-episode insights (scope=episode)
    - Persistent memory: cross-episode knowledge (scope=persistent)
    - Tag filtering: LLM can enable/disable tags to control what's shown
    - Adaptive thinking: LLM self-selects depth of reasoning per step
    """

    # Commands that accept a direction as follow-up input
    _DIRECTIONAL_COMMANDS = frozenset({
        "open", "close", "kick", "fight", "force",
        "zap", "throw", "fire", "untrap", "loot",
    })

    # Valid directions for compound actions
    _DIRECTIONS = frozenset({
        "north", "south", "east", "west",
        "northeast", "northwest", "southeast", "southwest",
        "up", "down",
    })

    # Load prompt from external file
    _PROMPT_FILE = Path(__file__).parent / "prompt.txt"
    _SYSTEM_PROMPT_SUFFIX = _PROMPT_FILE.read_text().strip()

    def __init__(self, client_factory: Any, prompt_builder: Any, config: Any):
        super().__init__(client_factory, prompt_builder)
        self.config = config
        braid_cfg = config.agent.braid

        # Initialize unified storage (SQLite with WAL for multi-worker support)
        self.storage = BraidStorage(Path(braid_cfg.db_path))
        self.max_memory_context = braid_cfg.get("max_persistent_context", 100)
        self.episode_number = self.storage.max_episode()
        self._enabled_tags: set[str] | None = None

        # Override history limit if specified
        if "max_text_history" in braid_cfg:
            self.prompt_builder.max_text_history = braid_cfg.max_text_history

        # Journal config
        self._log_full_prompt = braid_cfg.get("log_full_prompt", False)
        self._log_memory_details = braid_cfg.get("log_memory_details", False)

        # Step and token tracking (storage is stateless)
        self._step = 0
        self._request_start: float | None = None
        self._cumulative_input_tokens = 0
        self._cumulative_output_tokens = 0
        self._episode_input_tokens = 0
        self._episode_output_tokens = 0
        self._cumulative_llm_calls = 0
        self._episode_llm_calls = 0

        # Track memory update counts and details
        self._mem_adds = 0
        self._mem_removes = 0
        self._mem_episode_adds = 0
        self._mem_persistent_adds = 0
        self._mem_episode_removes = 0
        self._mem_persistent_removes = 0
        self._tag_changes = False
        self._added_entries: list[dict[str, Any]] = []
        self._removed_ids: list[str] = []

        # Multi-action queue state
        self._action_queue: list[str] = []
        self._queue_start_hp: int | None = None
        self._cautious_mode: bool = False
        self._last_glyphs: Any = None  # For cautious mode discovery detection
        self._batch_source: str | None = None  # Command that created batch
        self._batch_total: int = 0  # Original queue size
        self._batch_executed: int = 0  # Actions executed so far

        # Extended thinking preservation
        # Opus 4.5+ has native thinking block preservation, so skip manual injection
        model_id = str(config.client.model_id).lower()
        self._is_opus = "opus" in model_id
        preserve_cfg = braid_cfg.get("preserve_extended_thinking", False)
        if preserve_cfg and self._is_opus:
            # Opus handles thinking preservation natively - skip manual injection
            self._preserve_extended_thinking = False
        else:
            self._preserve_extended_thinking = preserve_cfg
        self._max_extended_thinking_chars = braid_cfg.get("max_extended_thinking_chars", 8000)
        self._last_extended_thinking: str | None = None

        # Compute helpers state
        self._enable_compute = braid_cfg.get("enable_compute_helpers", True)
        self._pending_compute: list[str] = []
        self._compute_result: str | None = None

        # Recent action history for prompt injection
        self._recent_actions: list[str] = []
        self._recent_action_outputs: list[str] = []  # Formatted action summaries
        self._max_recent_actions = 10

        # Extended thinking detection (for model-specific prompt formatting)
        self._thinking_budget = config.client.generate_kwargs.get("thinking_budget", 0)
        self._has_extended_thinking = self._is_opus or self._thinking_budget > 0

        # Current game turn (from blstats) for correcting actlog entries
        self._current_game_turn: int | None = None

    def build_system_prompt(self, env_instruction: str) -> str:
        """Append BRAID response format instructions to environment prompt."""
        return f"{env_instruction}\n\n{self._SYSTEM_PROMPT_SUFFIX}"

    def _extract_screen(self, obs: dict[str, Any]) -> str | None:
        """Extract ASCII screen from observation (NetHack tty_chars)."""
        tty_chars = None
        glyphs = None
        # NLE wraps raw observation in obs["obs"]
        raw_obs = obs.get("obs")
        if isinstance(raw_obs, dict):
            tty_chars = raw_obs.get("tty_chars")
            glyphs = raw_obs.get("glyphs")
        # Fallback: check top level
        if tty_chars is None:
            tty_chars = obs.get("tty_chars")
        if glyphs is None:
            glyphs = obs.get("glyphs")
        if tty_chars is None:
            return None
        try:
            # Fix NLE bug: walls in glyphs may not appear in tty_chars
            if glyphs is not None:
                from balrog.environments.nle.base import _fix_missing_walls
                tty_chars = _fix_missing_walls(tty_chars, glyphs)
            rows, cols = tty_chars.shape
            lines = ["".join(chr(tty_chars[i, j]) for j in range(cols)) for i in range(rows)]
            return "\n".join(lines)
        except (AttributeError, TypeError):
            return None

    def act(self, obs: dict[str, Any], prev_action: str | None = None) -> LLMResponse:
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        # Extract position info for tracking (available for both queued and LLM actions)
        pos_info = self._extract_position_info(obs)

        # Process any pending compute requests from last turn
        self._process_pending_compute(obs)

        # Check action queue first - return queued action if available and safe
        if self._action_queue:
            if self._should_abort_queue(obs):
                self.storage.log_error(
                    self.episode_number, self._step,
                    f"Queue aborted ({len(self._action_queue)} actions remaining)"
                )
                self._record_batch_summary(completed=False)
                self._clear_batch_state()
            else:
                # Check for newly discovered traps and re-plan if needed
                self._check_and_replan_for_traps(obs)
                self._step += 1
                # Log position for queued action
                if pos_info:
                    x, y, dlvl = pos_info
                    self.storage.log_position(self.episode_number, self._step, dlvl, x, y)
                action_response = self._pop_queued_action()
                # Log queued action
                self.storage.log_response(
                    episode=self.episode_number,
                    step=self._step,
                    action=action_response.completion,
                    latency_ms=0,
                    input_tokens=0,
                    output_tokens=0,
                    ep_input_tokens=self._episode_input_tokens,
                    ep_output_tokens=self._episode_output_tokens,
                    total_input_tokens=self._cumulative_input_tokens,
                    total_output_tokens=self._cumulative_output_tokens,
                    reasoning=action_response.reasoning,
                    cache_creation_tokens=0,
                    cache_read_tokens=0,
                    action_type="queued",
                    ep_llm_calls=self._episode_llm_calls,
                    total_llm_calls=self._cumulative_llm_calls,
                )
                return action_response

        # Queue empty but batch in progress - check for auto-continuation
        if self._batch_source in ("explore_room", "explore_corridor"):
            if self._auto_continue_exploration(obs):
                # Queue was refilled, process from queue
                self._step += 1
                if pos_info:
                    x, y, dlvl = pos_info
                    self.storage.log_position(self.episode_number, self._step, dlvl, x, y)
                action_response = self._pop_queued_action()
                self.storage.log_response(
                    episode=self.episode_number,
                    step=self._step,
                    action=action_response.completion,
                    latency_ms=0,
                    input_tokens=0,
                    output_tokens=0,
                    ep_input_tokens=self._episode_input_tokens,
                    ep_output_tokens=self._episode_output_tokens,
                    total_input_tokens=self._cumulative_input_tokens,
                    total_output_tokens=self._cumulative_output_tokens,
                    reasoning=action_response.reasoning,
                    cache_creation_tokens=0,
                    cache_read_tokens=0,
                    action_type="queued",
                    ep_llm_calls=self._episode_llm_calls,
                    total_llm_calls=self._cumulative_llm_calls,
                )
                return action_response
            else:
                # Exploration complete, record batch summary
                self._record_batch_summary(completed=True)
                self._clear_batch_state()

        self.prompt_builder.update_observation(obs)
        messages = self.prompt_builder.get_prompt()

        if messages:
            messages[-1].content = self._build_enhanced_prompt(messages[-1].content, obs)

        # Increment step and start timing
        self._step += 1
        self._request_start = time.perf_counter()

        # Log position for LLM action
        pos_dlvl: int | None = None
        if pos_info:
            px, py, pos_dlvl = pos_info
            self.storage.log_position(self.episode_number, self._step, pos_dlvl, px, py)

        # Log screen if available (include dlvl for monitor overlay)
        screen = self._extract_screen(obs)
        if screen:
            self.storage.log_screen(self.episode_number, self._step, screen, dlvl=pos_dlvl)

        # Log request
        prompt_chars = sum(len(getattr(m, "content", str(m))) for m in messages)
        obs_text = obs.get("text", {}).get("short_term_context", "")
        full_prompt = None
        if self._log_full_prompt:
            full_prompt = [{"role": getattr(m, "role", "unknown"), "content": getattr(m, "content", str(m))} for m in messages]
        self.storage.log_request(
            self.episode_number, self._step, len(messages), prompt_chars, obs_text, full_prompt
        )

        self._cumulative_llm_calls += 1
        self._episode_llm_calls += 1
        response = self.client.generate(messages)

        # Log SDK incremental prompt if using claude-sdk client
        if hasattr(self.client, "get_incremental_history"):
            self.storage.log_sdk_prompt(
                self.episode_number,
                self._step,
                sent_content=getattr(self.client, "_last_sent", "") or "",
                received_content=getattr(self.client, "_last_received", "") or "",
                conversation_history=self.client.get_incremental_history(),
            )

        parsed = self._parse_response(response)

        # If queue was populated, track HP for abort detection
        if self._action_queue:
            self._queue_start_hp = self._extract_hp(obs)

        return parsed

    def _build_enhanced_prompt(self, current_content: str, obs: dict[str, Any]) -> str:
        """Build prompt optimized for cache hits and minimal queries."""
        sections = []

        # Auto-inject status from blstats
        raw_obs = obs.get("obs", {})
        blstats = raw_obs.get("blstats") if isinstance(raw_obs, dict) else None
        if blstats is not None:
            from .compute.navigation import format_status
            self._current_game_turn = int(blstats[20])
            sections.append(f"[STATUS] {format_status(blstats)}")

        p_entries = self.storage.retrieve(
            tags=self._enabled_tags, scope=MemoryScope.PERSISTENT, limit=self.max_memory_context
        )
        e_entries = self.storage.retrieve(
            tags=self._enabled_tags, scope=MemoryScope.EPISODE,
            episode=self.episode_number, limit=self.max_memory_context
        )

        if p_entries:
            lines = [f"[{e.entry_id}] T:{e.source_step or '?'} (prio:{e.priority}) (tags: {e.tags}) {e.content}" for e in p_entries]
            header = f"PERSISTENT memory ({len(p_entries)}"
            if len(p_entries) >= self.max_memory_context:
                total = self.storage.count(tags=self._enabled_tags, scope=MemoryScope.PERSISTENT)
                hidden = total - len(p_entries)
                if hidden > 0:
                    header += f"+{hidden} hidden due to limit"
            header += "):"
            sections.append(f"{header}\n" + "\n".join(lines))

        if e_entries:
            lines = [f"[{e.entry_id}] @{e.source_step or '?'} (prio:{e.priority}) (tags: {e.tags}) {e.content}" for e in e_entries]
            header = f"EPISODE memory ({len(e_entries)}"
            if len(e_entries) >= self.max_memory_context:
                total = self.storage.count(
                    tags=self._enabled_tags, scope=MemoryScope.EPISODE, episode=self.episode_number
                )
                hidden = total - len(e_entries)
                if hidden > 0:
                    header += f"+{hidden} hidden due to limit"
            header += "):"
            sections.append(f"{header}\n" + "\n".join(lines))

        tag_info = self._compute_tag_summary(p_entries + e_entries)
        if tag_info:
            sections.append(tag_info)

        # Inject previous extended thinking if preserved
        if self._preserve_extended_thinking and self._last_extended_thinking:
            sections.append(f"PREVIOUS REASONING (from last turn):\n{self._last_extended_thinking}")

        # Inject compute results if available
        if self._compute_result:
            sections.append(self._compute_result)
            self._compute_result = None

        # Inject recent action history (model-specific format)
        if self._recent_action_outputs:
            if self._has_extended_thinking:
                # Extended thinking models: show compact action outputs only
                actions_str = " | ".join(self._recent_action_outputs)
                sections.append(f"RECENT ACTIONS: {actions_str}")
            else:
                # Non-thinking models (Haiku): show verbose action list
                actions_str = ", ".join(self._recent_actions)
                sections.append(f"YOUR LAST ACTIONS (recent first): {actions_str}")

        sections.append(current_content)

        return "\n\n".join(sections)

    def _compute_tag_summary(self, entries: list[MemoryEntry]) -> str:
        """Compute tag summary from already-retrieved entries."""
        if not entries:
            return ""
        tag_counts: dict[str, int] = {}
        for e in entries:
            for tag in e.tags.split(","):
                tag = tag.strip()
                if tag:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
        if not tag_counts:
            return ""
        tags_str = " ".join(f"{t}:{c}" for t, c in sorted(tag_counts.items()))

        if self._enabled_tags is None:
            filter_state = "enabled tags: all"
        elif not self._enabled_tags:
            filter_state = "enabled tags: none (all hidden)"
        else:
            filter_state = f"enabled tags: {','.join(sorted(self._enabled_tags))}"
        return f"[{filter_state}] [tags counters: {tags_str}]"


    def _parse_response(self, response: LLMResponse) -> LLMResponse:
        completion = response.completion
        elapsed = time.perf_counter() - self._request_start if self._request_start else 0
        self._request_start = None

        # Update token counters
        self._cumulative_input_tokens += response.input_tokens
        self._cumulative_output_tokens += response.output_tokens
        self._episode_input_tokens += response.input_tokens
        self._episode_output_tokens += response.output_tokens

        think_match = re.search(r"<think>(.*?)</think>", completion, re.DOTALL)
        reasoning = think_match.group(1).strip() if think_match else None

        # Reset memory update counters
        self._mem_adds = 0
        self._mem_removes = 0
        self._mem_episode_adds = 0
        self._mem_persistent_adds = 0
        self._mem_episode_removes = 0
        self._mem_persistent_removes = 0
        self._tag_changes = False
        self._added_entries = []
        self._removed_ids = []

        mem_match = re.search(r"<memory_updates>(.*?)</memory_updates>", completion, re.DOTALL)
        if mem_match:
            self._process_memory_updates(mem_match.group(1))

        # Parse compute requests
        if self._enable_compute:
            compute_match = re.search(r"<\|COMPUTE\|>(.*?)<\|END\|>", completion, re.DOTALL)
            if compute_match:
                self._pending_compute = [
                    line.strip()
                    for line in compute_match.group(1).strip().split("\n")
                    if line.strip()
                ]

        # Parse actions (single or multi)
        actions = self._parse_multi_actions(completion)
        action = actions[0]

        # Queue remaining actions if multi-action response
        if len(actions) > 1:
            self._action_queue = actions[1:]
            self._batch_source = "multi-action"
            self._batch_total = len(actions)
            self._batch_executed = 1  # First action executed this turn

        if reasoning:
            self.prompt_builder.update_reasoning(reasoning)

        # Log response
        self.storage.log_response(
            episode=self.episode_number,
            step=self._step,
            action=action,
            latency_ms=int(elapsed * 1000),
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            ep_input_tokens=self._episode_input_tokens,
            ep_output_tokens=self._episode_output_tokens,
            total_input_tokens=self._cumulative_input_tokens,
            total_output_tokens=self._cumulative_output_tokens,
            reasoning=reasoning,
            cache_creation_tokens=getattr(response, "cache_creation_tokens", 0) or 0,
            cache_read_tokens=getattr(response, "cache_read_tokens", 0) or 0,
            extended_thinking=getattr(response, "extended_thinking", None),
            action_type="multi" if len(actions) > 1 else "single",
            compute_requests=self._pending_compute if self._pending_compute else None,
            raw_completion=completion,
            ep_llm_calls=self._episode_llm_calls,
            total_llm_calls=self._cumulative_llm_calls,
        )

        # Store extended thinking for next turn if enabled
        if self._preserve_extended_thinking:
            ext = getattr(response, "extended_thinking", None)
            if ext:
                if len(ext) > self._max_extended_thinking_chars:
                    ext = ext[: self._max_extended_thinking_chars] + "\n[...truncated]"
                self._last_extended_thinking = ext
            else:
                self._last_extended_thinking = None

        # Log memory updates
        self.storage.log_memory_update(
            episode=self.episode_number,
            step=self._step,
            adds=self._mem_adds,
            removes=self._mem_removes,
            tag_changes=self._tag_changes,
            added_entries=self._added_entries if self._log_memory_details else None,
            removed_ids=self._removed_ids if self._log_memory_details else None,
            episode_adds=self._mem_episode_adds,
            persistent_adds=self._mem_persistent_adds,
            episode_removes=self._mem_episode_removes,
            persistent_removes=self._mem_persistent_removes,
        )

        # Record action in recent history (with formatted output for extended thinking models)
        self._record_action(action)
        self._record_action_output(actions, self._pending_compute if self._pending_compute else None)

        return response._replace(reasoning=reasoning or completion, completion=action)

    # Words that are unlikely to be valid actions
    _SKIP_WORDS = frozenset({
        "yes", "no", "true", "false", "the", "a", "an", "is", "are", "was", "were",
        "i", "you", "it", "this", "that", "should", "would", "could", "will", "can",
        "thinking", "action", "memory", "response", "format", "required", "optional",
    })

    def _fallback_action(self, completion: str) -> str:
        """Extract action when tags missing. Default to 'esc' if unparseable."""
        lines = completion.strip().split("\n")
        for line in reversed(lines):
            line = line.strip().lower()
            # Skip XML-like tags and known non-action prefixes
            if not line or line.startswith(("<", "thinking", "memory", "add:", "remove:", "enable", "disable", "reset")):
                continue
            # Get last word, strip punctuation
            words = line.split()
            if words:
                candidate = words[-1].rstrip(".,!?:;")
                if candidate and candidate not in self._SKIP_WORDS:
                    self.storage.log_error(
                        self.episode_number, self._step,
                        f"Fallback action '{candidate}' from: {line[:50]}"
                    )
                    return candidate
        self.storage.log_error(
            self.episode_number, self._step,
            f"No action found, defaulting to 'esc'. Response: {completion[:100]}"
        )
        return "esc"

    # --- Multi-action queue support ---

    def _is_interactive_prompt(self, text: str) -> bool:
        """Detect if game is waiting for specific input (not a queued action).

        Returns True only if the prompt requires input that can't be satisfied
        by the next queued action.
        """
        if not self._action_queue:
            # No queued actions - any interactive prompt needs LLM input
            if "In what direction" in text:
                return True
            if re.search(r"\[[a-zA-Z0-9\s\?\*\-]+\]\s*$", text):
                return True
            if "How many" in text:
                return True
            if re.search(r"(Really|Are you sure|Continue)\s*\?", text, re.IGNORECASE):
                return True
            return False

        next_action = self._action_queue[0]

        # Direction prompt - check if next queued action is a direction
        if "In what direction" in text:
            return not self._is_direction(next_action)

        # Choice brackets at end of message: [abc or ?*], [yn], [ynq], etc.
        # This handles inventory selection, yes/no prompts, etc.
        bracket_match = re.search(r"\[([a-zA-Z0-9\s\?\*\-]+)\]\s*$", text)
        if bracket_match:
            # Single letter/character queued actions are valid responses
            # (inventory letters, menu choices, y/n)
            if len(next_action) == 1:
                return False  # Let queue continue with single-char response
            return True

        # Quantity prompt
        if "How many" in text:
            return not next_action.isdigit()

        # Confirmation questions without brackets
        if re.search(r"(Really|Are you sure|Continue)\s*\?", text, re.IGNORECASE):
            return next_action.lower() not in ("y", "n", "yes", "no")

        return False

    def _is_direction(self, action: str) -> bool:
        """Check if action is a valid direction."""
        directions = {
            "north", "south", "east", "west",
            "northeast", "northwest", "southeast", "southwest",
            "n", "s", "e", "w", "ne", "nw", "se", "sw",
            "up", "down", ".", ">", "<"
        }
        return action.lower() in directions

    def _should_abort_queue(self, obs: dict[str, Any]) -> bool:
        """Check if action queue should be aborted based on observation."""
        if not self._action_queue:
            return False

        text = obs.get("text", {}).get("long_term_context", "")
        text_lower = text.lower()

        # Abort on interactive prompts
        if self._is_interactive_prompt(text):
            return True

        # Pet interactions are normal, don't abort
        pet_patterns = ["swap places", "your kitten", "your cat", "your dog", "your pony",
                        "moves out of your way", "gets out of your way"]
        is_pet_interaction = any(p in text_lower for p in pet_patterns)

        # Abort on combat indicators (but not pet-related)
        combat_patterns = ["hits", "misses", "bites", "attacks", "throws", "swings"]
        if any(p in text_lower for p in combat_patterns) and not is_pet_interaction:
            return True

        # Abort on significant HP drop (>20% since queue started)
        if self._queue_start_hp:
            current_hp = self._extract_hp(obs)
            if current_hp is not None and current_hp < self._queue_start_hp * 0.8:
                return True

        # Abort on danger indicators
        danger_patterns = ["cursed", "poisoned", "confused", "blind", "stuck", "paralyzed"]
        if any(p in text_lower for p in danger_patterns):
            return True

        # Abort on trap trigger (but not just seeing "trap" in text)
        trap_triggers = ["trigger a", "fall into", "step on a", "you are caught"]
        if any(p in text_lower for p in trap_triggers):
            return True

        # Cautious mode: abort on nearby map discovery (new glyphs appearing close to player)
        if self._cautious_mode and self._last_glyphs is not None:
            raw_obs = obs.get("obs", {})
            glyphs = raw_obs.get("glyphs") if isinstance(raw_obs, dict) else None
            if glyphs is not None:
                import numpy as np
                from nle import nethack
                # Check if any new non-stone tiles appeared (discovery)
                stone_glyph = nethack.GLYPH_CMAP_OFF + 0
                was_stone = self._last_glyphs == stone_glyph
                now_not_stone = glyphs != stone_glyph
                newly_revealed = was_stone & now_not_stone

                # Only abort if reveals are NEAR the player (within 3 tiles)
                # Distant reveals (e.g., "You hear a door open") are not dangerous
                pos_info = self._extract_position_info(obs)
                if np.any(newly_revealed) and pos_info:
                    px, py = pos_info[0], pos_info[1]
                    rows, cols = glyphs.shape
                    nearby_reveal = False
                    for dy in range(-3, 4):
                        for dx in range(-3, 4):
                            ny, nx = py + dy, px + dx
                            if 0 <= ny < rows and 0 <= nx < cols:
                                if newly_revealed[ny, nx]:
                                    nearby_reveal = True
                                    break
                        if nearby_reveal:
                            break

                    if nearby_reveal:
                        self.storage.log_error(
                            self.episode_number, self._step,
                            f"Cautious abort: {np.sum(newly_revealed)} tiles revealed nearby"
                        )
                        return True
                # Update snapshot
                self._last_glyphs = glyphs.copy()

        return False

    def _check_and_replan_for_traps(self, obs: dict[str, Any]) -> bool:
        """Check if new traps discovered and re-plan exploration if needed.

        Returns True if queue was re-planned, False otherwise.
        """
        if not self._action_queue or not self._batch_source:
            return False

        # Only re-plan for exploration helpers
        if self._batch_source not in ("explore_room", "explore_corridor"):
            return False

        raw_obs = obs.get("obs", {})
        if not isinstance(raw_obs, dict):
            return False

        glyphs = raw_obs.get("glyphs")
        if glyphs is None or self._last_glyphs is None:
            return False

        from nle import nethack

        # Check for newly discovered traps
        new_traps = []
        rows, cols = glyphs.shape
        for row in range(rows):
            for col in range(cols):
                glyph = int(glyphs[row, col])
                old_glyph = int(self._last_glyphs[row, col])
                # Trap just appeared (wasn't a trap before, is now)
                if nethack.glyph_is_trap(glyph) and not nethack.glyph_is_trap(old_glyph):
                    new_traps.append((col, row))

        if not new_traps:
            return False

        # New trap discovered - re-plan the exploration
        pos_info = self._extract_position_info(obs)
        if not pos_info:
            return False

        pos = (pos_info[0], pos_info[1])
        dlvl = pos_info[2]

        # Get visited tiles for this level
        visited = self.storage.get_visited_for_level(self.episode_number, dlvl)

        # Re-plan based on exploration type
        if self._batch_source == "explore_room":
            from .compute.navigation import detect_room, plan_room_exploration
            room = detect_room(glyphs, pos)
            if room:
                new_actions = plan_room_exploration(glyphs, room, pos, visited=visited)
                if new_actions:
                    self._action_queue = list(new_actions)
                    self._batch_total = len(new_actions)
                    self._batch_executed = 0
                    self.storage.log_error(
                        self.episode_number, self._step,
                        f"Re-planned room exploration around {len(new_traps)} new trap(s)"
                    )
                    return True

        elif self._batch_source == "explore_corridor":
            from .compute.navigation import detect_corridor, plan_corridor_exploration
            corridor = detect_corridor(glyphs, pos)
            if corridor:
                new_actions = plan_corridor_exploration(glyphs, corridor, pos, visited=visited)
                if new_actions:
                    self._action_queue = list(new_actions)
                    self._batch_total = len(new_actions)
                    self._batch_executed = 0
                    self.storage.log_error(
                        self.episode_number, self._step,
                        f"Re-planned corridor exploration around {len(new_traps)} new trap(s)"
                    )
                    return True

        return False

    def _extract_hp(self, obs: dict[str, Any]) -> int | None:
        """Extract current HP from observation."""
        # Try blstats first (raw NLE observation)
        raw_obs = obs.get("obs")
        if isinstance(raw_obs, dict):
            blstats = raw_obs.get("blstats")
            if blstats is not None:
                try:
                    # blstats[10] is typically HP in NLE
                    return int(blstats[10])
                except (IndexError, TypeError, ValueError):
                    pass
        # Fallback: parse from text
        text = obs.get("text", {}).get("short_term_context", "")
        hp_match = re.search(r"HP:(\d+)", text)
        if hp_match:
            return int(hp_match.group(1))
        return None

    def _extract_position_info(
        self, obs: dict[str, Any]
    ) -> tuple[int, int, int] | None:
        """Extract (x, y, dlvl) from observation blstats."""
        raw_obs = obs.get("obs")
        if isinstance(raw_obs, dict):
            blstats = raw_obs.get("blstats")
            if blstats is not None:
                try:
                    x = int(blstats[0])
                    y = int(blstats[1])
                    dlvl = int(blstats[12])
                    return (x, y, dlvl)
                except (IndexError, TypeError, ValueError):
                    pass
        return None

    def _process_pending_compute(self, obs: dict[str, Any]) -> None:
        """Execute pending compute requests using current observation."""
        if not self._pending_compute:
            return

        from .compute.navigation import (
            distance,
            get_position,
            nearest,
            pathfind,
        )

        raw_obs = obs.get("obs", {})
        glyphs = raw_obs.get("glyphs") if isinstance(raw_obs, dict) else None
        blstats = raw_obs.get("blstats") if isinstance(raw_obs, dict) else None
        tty_chars = raw_obs.get("tty_chars") if isinstance(raw_obs, dict) else None

        if glyphs is None or blstats is None:
            self._compute_result = "[COMPUTE] ERROR: No glyph data available"
            self._pending_compute = []
            return

        pos = get_position(blstats)
        results = []

        for request in self._pending_compute:
            if request.startswith("distance:"):
                match = re.match(r"distance:\s*@(\d+),(\d+)\s*->\s*@(\d+),(\d+)", request)
                if match:
                    x1, y1, x2, y2 = map(int, match.groups())
                    d = distance(x1, y1, x2, y2)
                    results.append(f"distance: @{x1},{y1} -> @{x2},{y2} = {d} tiles")
                else:
                    results.append("distance: PARSE ERROR")

            elif request.startswith("nearest:"):
                feature = request.split(":", 1)[1].strip()
                result = nearest(glyphs, pos, feature)
                if result:
                    x, y, d = result
                    results.append(f"nearest: {feature} = @{x},{y} ({d} tiles)")
                else:
                    results.append(f"nearest: {feature} = NOT FOUND in explored areas")

            elif request.startswith("pathfind:"):
                from .compute.navigation import find_monster_positions
                match = re.match(r"pathfind:\s*@(\d+),(\d+)\s*->\s*@(\d+),(\d+)", request)
                if match:
                    x1, y1, x2, y2 = map(int, match.groups())
                    # Include monster/pet positions as walkable (can attack/swap)
                    extra = {pos} | find_monster_positions(glyphs)
                    path = pathfind(glyphs, (x1, y1), (x2, y2), extra_walkable=extra)
                    if path:
                        dirs = " ".join(path)
                        results.append(
                            f"pathfind: @{x1},{y1} -> @{x2},{y2} = {dirs} ({len(path)} moves)"
                        )
                    else:
                        results.append(
                            f"pathfind: @{x1},{y1} -> @{x2},{y2} = NO PATH (unexplored/blocked)"
                        )
                else:
                    results.append("pathfind: PARSE ERROR")

            elif request.startswith("travel_to:"):
                # Absolute: travel_to: @45,12
                # Relative: travel_to: +3,-2 (x+3, y-2 from current position)
                from .compute.navigation import find_monster_positions
                abs_match = re.match(r"travel_to:\s*@(\d+),(\d+)", request)
                rel_match = re.match(r"travel_to:\s*([+-]?\d+),([+-]?\d+)", request)
                if abs_match:
                    gx, gy = map(int, abs_match.groups())
                elif rel_match:
                    dx, dy = map(int, rel_match.groups())
                    gx, gy = pos[0] + dx, pos[1] + dy
                else:
                    results.append("travel_to: PARSE ERROR (use @x,y or +dx,dy)")
                    continue

                # Include monster/pet positions as walkable (can attack/swap)
                extra = {pos} | find_monster_positions(glyphs)
                path = pathfind(glyphs, pos, (gx, gy), extra_walkable=extra)
                if path:
                    self._action_queue.extend(path)
                    self._queue_start_hp = self._extract_hp(obs)
                    self._batch_source = f"travel_to(@{gx},{gy})"
                    self._batch_total = len(path)
                    self._batch_executed = 0
                    results.append(f"travel_to: @{gx},{gy} = EXECUTING {len(path)} moves (auto)")
                else:
                    results.append(f"travel_to: @{gx},{gy} = NO PATH (unexplored/blocked)")

            elif request.startswith("travel:"):
                # Direction-based relative travel: travel: north 5, travel: NE 3
                from .compute.navigation import DIRS, find_monster_positions
                dir_match = re.match(
                    r"travel:\s*(north|south|east|west|northeast|northwest|southeast|southwest|N|S|E|W|NE|NW|SE|SW)\s+(\d+)",
                    request, re.IGNORECASE
                )
                if dir_match:
                    dir_name, dist_str = dir_match.groups()
                    dist = int(dist_str)
                    # Normalize direction name
                    dir_map = {
                        "n": "north", "s": "south", "e": "east", "w": "west",
                        "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest"
                    }
                    dir_name = dir_map.get(dir_name.lower(), dir_name.lower())
                    if dir_name in DIRS:
                        dy, dx = DIRS[dir_name]
                        gx, gy = pos[0] + dx * dist, pos[1] + dy * dist
                        # Include monster/pet positions as walkable (can attack/swap)
                        extra = {pos} | find_monster_positions(glyphs)
                        path = pathfind(glyphs, pos, (gx, gy), extra_walkable=extra)
                        if path:
                            self._action_queue.extend(path)
                            self._queue_start_hp = self._extract_hp(obs)
                            self._batch_source = f"travel({dir_name},{dist})"
                            self._batch_total = len(path)
                            self._batch_executed = 0
                            results.append(f"travel: {dir_name} {dist} -> @{gx},{gy} = EXECUTING {len(path)} moves (auto)")
                        else:
                            results.append(f"travel: {dir_name} {dist} -> @{gx},{gy} = NO PATH")
                    else:
                        results.append(f"travel: UNKNOWN DIRECTION '{dir_name}'")
                else:
                    results.append("travel: PARSE ERROR (use 'travel: north 5' or 'travel: NE 3')")

            elif request.strip() == "scan_monsters":
                from .compute.navigation import scan_monsters
                if tty_chars is not None:
                    results.append(f"scan_monsters: {scan_monsters(glyphs, tty_chars, pos)}")
                else:
                    results.append("scan_monsters: ERROR (no tty_chars)")

            elif request.strip() == "scan_items":
                from .compute.navigation import scan_items
                if tty_chars is not None:
                    results.append(f"scan_items: {scan_items(glyphs, tty_chars, pos)}")
                else:
                    results.append("scan_items: ERROR (no tty_chars)")

            elif request.strip() == "scan_traps":
                from .compute.navigation import scan_traps
                if tty_chars is not None:
                    results.append(f"scan_traps: {scan_traps(glyphs, tty_chars, pos)}")
                else:
                    results.append("scan_traps: ERROR (no tty_chars)")

            elif request.strip() == "unexplored":
                from .compute.navigation import find_unexplored
                results.append(f"unexplored: {find_unexplored(glyphs, pos)}")

            elif request.strip() == "exits":
                from .compute.navigation import find_exits
                results.append(f"exits: {find_exits(glyphs, pos)}")

            elif request.strip() in ("explore_room", "explore_room:cautious"):
                from .compute.navigation import detect_room, plan_room_exploration
                cautious = request.strip().endswith(":cautious")
                room = detect_room(glyphs, pos)
                if room:
                    # Get visited tiles for this level
                    dlvl = int(blstats[12])
                    visited = self.storage.get_visited_for_level(self.episode_number, dlvl)
                    actions = plan_room_exploration(glyphs, room, pos, visited=visited)
                    if actions:
                        self._action_queue.extend(actions)
                        self._queue_start_hp = self._extract_hp(obs)
                        self._cautious_mode = cautious
                        self._batch_source = "explore_room"
                        self._batch_total = len(actions)
                        self._batch_executed = 0
                        if cautious:
                            self._last_glyphs = glyphs.copy()
                        mode_str = " (cautious)" if cautious else ""
                        results.append(f"explore_room{mode_str}: EXECUTING {len(actions)} actions (auto)")
                    else:
                        results.append("explore_room: NO ACTIONS (already at perimeter?)")
                else:
                    results.append("explore_room: NOT IN ROOM (try explore_corridor)")

            elif request.strip() in ("explore_corridor", "explore_corridor:cautious"):
                from .compute.navigation import detect_corridor, plan_corridor_exploration
                cautious = request.strip().endswith(":cautious")
                corridor = detect_corridor(glyphs, pos)
                if corridor:
                    # Get visited tiles for this level
                    dlvl = int(blstats[12])
                    visited = self.storage.get_visited_for_level(self.episode_number, dlvl)
                    actions = plan_corridor_exploration(glyphs, corridor, pos, visited=visited)
                    if actions:
                        self._action_queue.extend(actions)
                        self._queue_start_hp = self._extract_hp(obs)
                        self._cautious_mode = cautious
                        self._batch_source = "explore_corridor"
                        self._batch_total = len(actions)
                        self._batch_executed = 0
                        if cautious:
                            self._last_glyphs = glyphs.copy()
                        mode_str = " (cautious)" if cautious else ""
                        results.append(f"explore_corridor{mode_str}: EXECUTING {len(actions)} actions (auto)")
                    else:
                        results.append("explore_corridor: NO ACTIONS")
                else:
                    results.append("explore_corridor: NOT IN CORRIDOR (try explore_room)")

            else:
                results.append(f"UNKNOWN: {request}")

        self._compute_result = "[COMPUTE]\n" + "\n".join(results)

        # Log compute helper usage for debugging
        self.storage.log_compute(
            episode=self.episode_number,
            step=self._step,
            requests=self._pending_compute,
            results=results,
        )
        self._pending_compute = []

    def _expand_compound_action(self, action: str) -> list[str]:
        """Expand compound action like 'open north' into ['open', 'north']."""
        parts = action.lower().split()
        if len(parts) == 2:
            cmd, direction = parts
            if cmd in self._DIRECTIONAL_COMMANDS and direction in self._DIRECTIONS:
                return [cmd, direction]
        return [action]

    def _parse_multi_actions(self, completion: str) -> list[str]:
        """Parse single action or multi-action block from completion."""
        # Check for <|ACTIONS|>...<|END|> block (multi-action)
        multi_match = re.search(r"<\|ACTIONS\|>(.*?)<\|END\|>", completion, re.DOTALL)
        if multi_match:
            actions = [a.strip() for a in multi_match.group(1).strip().split("\n") if a.strip()]
        elif (single_match := re.search(r"<\|ACTION\|>(.*?)<\|END\|>", completion, re.DOTALL)):
            actions = [single_match.group(1).strip()]
        elif self._pending_compute:
            # COMPUTE found but no ACTION - use "wait" as safe no-op
            # COMPUTE will be processed next turn and queue actions
            actions = ["wait"]
        else:
            actions = [self._fallback_action(completion)]

        # Expand compound actions (e.g., "open north" -> ["open", "north"])
        expanded: list[str] = []
        for action in actions:
            expanded.extend(self._expand_compound_action(action))
        return expanded if expanded else [self._fallback_action(completion)]

    def _record_action(self, action: str) -> None:
        """Record action in recent history (most recent first)."""
        self._recent_actions.insert(0, action)
        if len(self._recent_actions) > self._max_recent_actions:
            self._recent_actions.pop()

    def _record_action_output(self, actions: list[str], compute_requests: list[str] | None) -> None:
        """Record formatted action output for extended thinking model prompts."""
        if compute_requests:
            output = f"COMPUTE:{compute_requests[0][:30]}"
        elif len(actions) > 1:
            abbrev = ",".join(actions[:5])
            output = f"ACTIONS:[{abbrev}...]" if len(actions) > 5 else f"ACTIONS:[{abbrev}]"
        else:
            output = f"ACTION:{actions[0]}"

        self._recent_action_outputs.insert(0, output)
        if len(self._recent_action_outputs) > self._max_recent_actions:
            self._recent_action_outputs.pop()

    def _record_batch_summary(self, completed: bool) -> None:
        """Record batch execution summary to action history."""
        if not self._batch_source:
            return
        status = "done" if completed else "aborted"
        summary = f"{self._batch_source}: {self._batch_executed}/{self._batch_total} {status}"
        self._recent_action_outputs.insert(0, summary)
        if len(self._recent_action_outputs) > self._max_recent_actions:
            self._recent_action_outputs.pop()

    def _clear_batch_state(self) -> None:
        """Clear all batch/queue tracking state."""
        self._action_queue.clear()
        self._queue_start_hp = None
        self._cautious_mode = False
        self._last_glyphs = None
        self._batch_source = None
        self._batch_total = 0
        self._batch_executed = 0

    def _pop_queued_action(self) -> LLMResponse:
        """Return next queued action without LLM call."""
        action = self._action_queue.pop(0)
        remaining = len(self._action_queue)
        self._batch_executed += 1

        # Record batch summary and clear state when exhausted
        if not self._action_queue:
            self._record_batch_summary(completed=True)
            # Note: Don't clear batch state yet - _auto_continue_exploration may refill queue
            # The state will be cleared in act() after checking for continuation

        return LLMResponse(
            model_id="queued",
            completion=action,
            stop_reason="queued",
            input_tokens=0,
            output_tokens=0,
            reasoning=f"[Queued: {remaining} remaining]",
        )

    def _auto_continue_exploration(self, obs: dict[str, Any]) -> bool:
        """Check if exploration should auto-continue after queue empties.

        When explore_room or explore_corridor completes, new tiles may have
        become visible. Re-plan and continue if there's more to explore.

        Returns True if queue was refilled, False otherwise.
        """
        if self._action_queue:  # Queue not empty
            return False

        if self._batch_source not in ("explore_room", "explore_corridor"):
            return False

        # Get current observation data
        raw_obs = obs.get("obs", {})
        if not isinstance(raw_obs, dict):
            return False

        glyphs = raw_obs.get("glyphs")
        blstats = raw_obs.get("blstats")
        if glyphs is None or blstats is None:
            return False

        from .compute.navigation import (
            detect_corridor,
            detect_room,
            get_position,
            plan_corridor_exploration,
            plan_room_exploration,
        )

        pos = get_position(blstats)
        dlvl = int(blstats[12])
        visited = self.storage.get_visited_for_level(self.episode_number, dlvl)

        new_actions: list[str] = []

        if self._batch_source == "explore_room":
            room = detect_room(glyphs, pos)
            if room:
                new_actions = plan_room_exploration(glyphs, room, pos, visited=visited)

        elif self._batch_source == "explore_corridor":
            corridor = detect_corridor(glyphs, pos)
            if corridor:
                new_actions = plan_corridor_exploration(glyphs, corridor, pos, visited=visited)

        if new_actions:
            self._action_queue = list(new_actions)
            self._batch_total += len(new_actions)
            # Don't reset _batch_executed - keep cumulative count
            self.storage.log_error(
                self.episode_number, self._step,
                f"Auto-continuing {self._batch_source}: {len(new_actions)} more actions"
            )
            return True

        return False

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

                    # Auto-correct turn in actlog entries (LLM often gets it wrong)
                    if "actlog" in tags and self._current_game_turn is not None:
                        content = re.sub(r"^T\d+:", f"T{self._current_game_turn}:", content)

                    if not tags or not content:
                        continue

                    scope = MemoryScope.EPISODE if scope_str == "episode" else MemoryScope.PERSISTENT

                    entry_id = self.storage.store(
                        MemoryEntry(
                            tags=tags,
                            content=content,
                            scope=scope,
                            priority=prio,
                            source_episode=self.episode_number,
                            source_step=self._step,
                        )
                    )
                    self._mem_adds += 1
                    if scope == MemoryScope.EPISODE:
                        self._mem_episode_adds += 1
                    else:
                        self._mem_persistent_adds += 1
                    if self._log_memory_details:
                        self._added_entries.append({
                            "id": entry_id,
                            "scope": scope_str,
                            "tags": tags,
                            "prio": prio,
                            "content": content,
                        })
            except (json.JSONDecodeError, ValueError):
                self.storage.log_error(self.episode_number, self._step, "Failed to parse memory additions")

        # Parse removals
        remove_match = re.search(r"remove:\s*\[(.+?)\]", mem_text, re.DOTALL)
        if remove_match:
            try:
                removals = json.loads(f"[{remove_match.group(1)}]")
                for entry_id in removals:
                    if isinstance(entry_id, str):
                        removed_scope = self.storage.remove(entry_id)
                        if removed_scope:
                            self._mem_removes += 1
                            if removed_scope == "episode":
                                self._mem_episode_removes += 1
                            else:
                                self._mem_persistent_removes += 1
                            if self._log_memory_details:
                                self._removed_ids.append(entry_id)
            except json.JSONDecodeError:
                self.storage.log_error(self.episode_number, self._step, "Failed to parse memory removals")

        # Parse enable_tags
        enable_match = re.search(r"enable_tags:\s*\[(.+?)\]", mem_text, re.DOTALL)
        if enable_match:
            try:
                tags = json.loads(f"[{enable_match.group(1)}]")
                tags = {str(lbl).lower().strip() for lbl in tags if isinstance(lbl, str)}
                if tags:
                    if self._enabled_tags is not None:
                        self._enabled_tags |= tags
                    self._tag_changes = True
            except json.JSONDecodeError:
                self.storage.log_error(self.episode_number, self._step, "Failed to parse enable_tags")

        # Parse disable_tags
        disable_match = re.search(r"disable_tags:\s*\[(.+?)\]", mem_text, re.DOTALL)
        if disable_match:
            try:
                tags = json.loads(f"[{disable_match.group(1)}]")
                tags = {str(lbl).lower().strip() for lbl in tags if isinstance(lbl, str)}
                if tags:
                    if self._enabled_tags is None:
                        all_tags = self.storage.all_tags()
                        self._enabled_tags = all_tags - tags
                    else:
                        self._enabled_tags -= tags
                    self._tag_changes = True
            except json.JSONDecodeError:
                self.storage.log_error(self.episode_number, self._step, "Failed to parse disable_tags")

        # Parse reset_tags
        reset_match = re.search(r"reset_tags:\s*(true|false)", mem_text, re.IGNORECASE)
        if reset_match and reset_match.group(1).lower() == "true":
            self._enabled_tags = None
            self._tag_changes = True

    def reset(self) -> None:
        """Reset for new episode."""
        super().reset()
        self.episode_number += 1
        self._step = 0
        self._enabled_tags = None
        self._episode_input_tokens = 0
        self._episode_output_tokens = 0
        self._episode_llm_calls = 0
        self._action_queue.clear()
        self._queue_start_hp = None
        self._cautious_mode = False
        self._last_glyphs = None
        self._batch_source = None
        self._batch_total = 0
        self._batch_executed = 0
        self._last_extended_thinking = None
        self._recent_actions.clear()
        self._recent_action_outputs.clear()
        self.storage.log_reset(self.episode_number, str(self.config.client.model_id))

    def on_episode_end(self) -> None:
        """Called when episode ends - close SDK session if active.

        This allows ClaudeSDKWrapper to close its session and start fresh
        for the next episode, maintaining clean context boundaries.
        """
        if hasattr(self.client, "close_session"):
            self.client.close_session()
