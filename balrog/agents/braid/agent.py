import logging
import re
import time
from pathlib import Path
from typing import Any

from balrog.agents.base import BaseAgent
from balrog.client import LLMResponse

from .storage import BraidStorage, MemoryEntry, MemoryScope

logger = logging.getLogger(__name__)


class BRAIDAgent(BaseAgent):
    """BALROG Recurrent Agentic Iterative Dungeoneer.

    An agent that learns from experience through unified memory:
    - Episode memory: per-episode insights (scope=episode)
    - Persistent memory: cross-episode knowledge (scope=persistent)
    - Claude SDK tools: memory, navigation, and action tools for LLM interaction
    """

    # Load prompt from external file (tool-based format)
    _PROMPT_FILE = Path(__file__).parent / "prompt_tools.txt"
    _SYSTEM_PROMPT_SUFFIX = _PROMPT_FILE.read_text().strip()

    def __init__(self, client_factory: Any, prompt_builder: Any, config: Any):
        # BRAID requires Claude Tools client for tool-based interactions
        client_name = config.client.client_name.lower()
        if client_name != "claude-tools":
            raise ValueError(
                f"BRAIDAgent requires client_name='claude-tools', got '{client_name}'. "
                f"Supported models: claude-haiku-4-5, claude-sonnet-4, claude-opus-4-5"
            )

        super().__init__(client_factory, prompt_builder)
        self.config = config
        braid_cfg = config.agent.braid

        # Initialize unified storage (SQLite with WAL for multi-worker support)
        self.storage = BraidStorage(Path(braid_cfg.db_path))
        self.max_memory_context = braid_cfg.get("max_persistent_context", 100)
        self.episode_number = self.storage.max_episode()

        # Create MCP server with BRAID tools and attach to client
        from .tools import create_braid_mcp_server

        self._mcp_server = create_braid_mcp_server(self.storage)
        if hasattr(self.client, "set_mcp_server"):
            self.client.set_mcp_server(self._mcp_server)
            logger.info("BRAID MCP server attached to client")

        # Override history limit if specified
        if "max_text_history" in braid_cfg:
            self.prompt_builder.max_text_history = braid_cfg.max_text_history

        # Journal config
        self._log_full_prompt = braid_cfg.get("log_full_prompt", False)

        # Step and token tracking (storage is stateless)
        self._step = 0
        self._request_start: float | None = None
        self._cumulative_input_tokens = 0
        self._cumulative_output_tokens = 0
        self._episode_input_tokens = 0
        self._episode_output_tokens = 0
        self._cumulative_llm_calls = 0
        self._episode_llm_calls = 0

        # Multi-action queue state
        self._action_queue: list[str] = []
        self._queue_start_hp: int | None = None
        self._queue_start_hunger: int | None = None  # Hunger state when queue started (0=Satiated, 2=Hungry, etc.)
        self._cautious_mode: bool = False
        self._last_glyphs: Any = None  # For cautious mode discovery detection
        self._batch_source: str | None = None  # Command that created batch
        self._batch_total: int = 0  # Original queue size
        self._batch_executed: int = 0  # Actions executed so far
        self._max_explore_actions: int = 100  # Safety limit for explore_room/corridor

        # Current game turn (from blstats) for correcting actlog entries
        self._current_game_turn: int | None = None

        # Queue abort notification (shown to LLM on next prompt)
        self._queue_abort_msg: str | None = None

        # Messages accumulated during queue execution
        self._queue_messages: list[str] = []

        # Memory refresh interval (include memory in system prompt every N LLM calls)
        self._memory_refresh_interval: int = 100

        # Level change tracking for branch announcements
        self._last_dungeon_num: int | None = None
        self._last_dlvl: int | None = None

    def build_system_prompt(self, env_instruction: str) -> str:
        """Append BRAID response format instructions to environment prompt."""
        return f"{env_instruction}\n\n{self._SYSTEM_PROMPT_SUFFIX}"

    def _extract_message(self, obs: dict[str, Any]) -> str:
        """Extract game message from observation."""
        raw_obs = obs.get("obs")
        if isinstance(raw_obs, dict):
            msg = raw_obs.get("text_message", "")
            if msg:
                return str(msg).strip()
        return ""

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

        # Check action queue first - return queued action if available and safe
        if self._action_queue:
            # Capture message from previous action (this observation is result of last queued action)
            queue_msg = self._extract_message(obs)
            if queue_msg:
                self._queue_messages.append(queue_msg)

            abort_reason = self._should_abort_queue(obs)
            if abort_reason:
                remaining = len(self._action_queue)
                self.storage.log_error(
                    self.episode_number, self._step,
                    f"Queue aborted ({remaining} actions remaining): {abort_reason}"
                )
                # Store abort message for next LLM prompt (includes accumulated messages)
                self._queue_abort_msg = self._format_queue_summary(
                    aborted=True, reason=abort_reason, remaining=remaining
                )
                self._clear_batch_state()
            else:
                # Check for newly discovered traps and re-plan if needed
                self._check_and_replan_for_traps(obs)
                self._step += 1
                # Log position and screen for queued action (for Web UI)
                queued_dlvl: int | None = None
                if pos_info:
                    x, y, dungeon_num, queued_dlvl = pos_info
                    self.storage.log_position(self.episode_number, self._step, dungeon_num, queued_dlvl, x, y)
                screen = self._extract_screen(obs)
                if screen:
                    self.storage.log_screen(self.episode_number, self._step, screen, dlvl=queued_dlvl)
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
                cont_dlvl: int | None = None
                if pos_info:
                    x, y, dungeon_num, cont_dlvl = pos_info
                    self.storage.log_position(self.episode_number, self._step, dungeon_num, cont_dlvl, x, y)
                screen = self._extract_screen(obs)
                if screen:
                    self.storage.log_screen(self.episode_number, self._step, screen, dlvl=cont_dlvl)
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
                # Exploration complete - capture summary before clearing
                if self._queue_messages:
                    self._queue_abort_msg = self._format_queue_summary(aborted=False)
                self._clear_batch_state()

        # If queue just finished (batch was in progress but queue is now empty), capture summary
        if self._batch_total > 0 and not self._action_queue and not self._queue_abort_msg:
            # Capture final message from this observation
            final_msg = self._extract_message(obs)
            if final_msg:
                self._queue_messages.append(final_msg)
            if self._queue_messages:
                self._queue_abort_msg = self._format_queue_summary(aborted=False)
            self._clear_batch_state()

        self.prompt_builder.update_observation(obs)
        messages = self.prompt_builder.get_prompt()

        if messages:
            # Enhance user message with status + observation
            messages[-1].content = self._build_enhanced_prompt(messages[-1].content, obs)

            # Inject memory into system prompt on first call and every N LLM calls
            include_memory = (
                self._episode_llm_calls == 0  # First call this episode
                or (self._memory_refresh_interval > 0
                    and self._episode_llm_calls % self._memory_refresh_interval == 0)
            )
            if include_memory and messages[0].role == "system":
                memory_context = self._build_memory_context()
                if memory_context:
                    messages[0].content = f"{messages[0].content}\n\n[CURRENT MEMORY]\n{memory_context}"

        # Increment step and start timing
        self._step += 1
        self._request_start = time.perf_counter()

        # Log position for LLM action
        pos_dlvl: int | None = None
        if pos_info:
            px, py, dungeon_num, pos_dlvl = pos_info
            self.storage.log_position(self.episode_number, self._step, dungeon_num, pos_dlvl, px, py)

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

        # Set tool context before generate (tools need current obs, storage, episode, step)
        from .tools import get_pending_actions, set_tool_context

        set_tool_context(obs, self.storage, self.episode_number, self._step)

        response = self.client.generate(messages)

        # Get pending actions from tools (auto_explore, game_action populate these)
        # Returns (actions, error_message) - error_message is set if multiple action tools called
        tool_actions, action_error = get_pending_actions()
        if action_error:
            # Multiple action tools called - reject all, notify on next turn
            self._queue_abort_msg = action_error
            logger.warning(f"Multiple action tools violation: {action_error}")
        elif tool_actions:
            self._action_queue.extend(tool_actions)
            if not self._batch_source:
                self._batch_source = "tool_action"
                self._batch_total = len(tool_actions)
                self._batch_executed = 0
            logger.debug(f"Tools queued {len(tool_actions)} actions")

        # Extract and save tool calls from SDK
        if hasattr(self.client, "get_tool_calls"):
            tool_calls = self.client.get_tool_calls()
            if tool_calls:
                logger.info(f"SDK tool calls: {[tc.get('name') for tc in tool_calls]}")
            for tc in tool_calls:
                tool_name = tc.get("name", "")
                tool_input = tc.get("input", {})
                tool_result = tc.get("result")
                # Save TodoWrite todos
                if tool_name == "TodoWrite":
                    todos = tool_input.get("todos", [])
                    if todos:
                        self.storage.save_todos(self.episode_number, self._step, todos)
                # Log all tool calls to storage for Web UI
                import json
                self.storage.log_tool_call(
                    episode=self.episode_number,
                    step=self._step,
                    tool_name=tool_name,
                    args=json.dumps(tool_input) if tool_input else None,
                    result=str(tool_result)[:500] if tool_result else None,
                )

        # Log SDK incremental prompt (SDK-only, always available)
        self.storage.log_sdk_prompt(
            self.episode_number,
            self._step,
            sent_content=getattr(self.client, "_last_sent", "") or "",
            received_content=getattr(self.client, "_last_received", "") or "",
            conversation_history=self.client.get_incremental_history(),
        )

        parsed = self._parse_response(response)

        # If queue was populated, track HP and hunger for abort detection
        if self._action_queue:
            self._queue_start_hp = self._extract_hp(obs)
            self._queue_start_hunger = self._extract_hunger(obs)

        return parsed

    def _build_memory_context(self) -> str:
        """Build formatted memory context for system prompt injection."""
        sections = []

        p_entries = self.storage.retrieve(
            scope=MemoryScope.PERSISTENT, limit=self.max_memory_context
        )
        e_entries = self.storage.retrieve(
            scope=MemoryScope.EPISODE,
            episode=self.episode_number, limit=self.max_memory_context
        )

        if p_entries:
            lines = [f"[{e.entry_id}] T:{e.source_step or '?'} (prio:{e.priority}) (tags: {e.tags}) {e.content}" for e in p_entries]
            header = f"PERSISTENT memory ({len(p_entries)}"
            if len(p_entries) >= self.max_memory_context:
                total = self.storage.count(scope=MemoryScope.PERSISTENT)
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
                    scope=MemoryScope.EPISODE, episode=self.episode_number
                )
                hidden = total - len(e_entries)
                if hidden > 0:
                    header += f"+{hidden} hidden due to limit"
            header += "):"
            sections.append(f"{header}\n" + "\n".join(lines))

        tag_info = self._compute_tag_summary(p_entries + e_entries)
        if tag_info:
            sections.append(tag_info)

        return "\n\n".join(sections)

    def _build_enhanced_prompt(self, current_content: str, obs: dict[str, Any]) -> str:
        """Build user prompt with status and observation (no memory - that goes in system)."""
        sections = []

        # Auto-inject status from blstats
        raw_obs = obs.get("obs", {})
        blstats = raw_obs.get("blstats") if isinstance(raw_obs, dict) else None
        if blstats is not None:
            from .compute.navigation import format_status, get_branch_name
            self._current_game_turn = int(blstats[20])
            sections.append(f"[STATUS] {format_status(blstats)}")

            # Check for level/branch change and announce
            dungeon_num = int(blstats[23])
            dlvl = int(blstats[12])
            level_changed = (
                self._last_dungeon_num is not None
                and (dungeon_num != self._last_dungeon_num or dlvl != self._last_dlvl)
            )
            if level_changed:
                branch_name = get_branch_name(dungeon_num)
                if dungeon_num != self._last_dungeon_num:
                    # Branch change (e.g., entering Mines or returning to Dungeon)
                    sections.append(f"[LEVEL AND DUNGEON BRANCH CHANGE] Entered {branch_name}, level {dlvl}.")
                else:
                    # Same branch, different level
                    sections.append(f"[LEVEL CHANGE] Now on {branch_name} level {dlvl}.")

            # Update tracking
            self._last_dungeon_num = dungeon_num
            self._last_dlvl = dlvl

        # Inject queue abort notification if queue was just aborted
        if self._queue_abort_msg:
            sections.append(self._queue_abort_msg)
            self._queue_abort_msg = None  # Clear after use

        sections.append(current_content)

        sections.append("Think carefully about the current information, todos and goals, the result of your last actions, available tools and actions, before issuing your next action(s).")

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
        return f"[tag counts: {tags_str}]"


    def _parse_response(self, response: LLMResponse) -> LLMResponse:
        """Process LLM response. Actions come from tools, not XML parsing."""
        completion = response.completion
        elapsed = time.perf_counter() - self._request_start if self._request_start else 0
        self._request_start = None

        # Update token counters
        self._cumulative_input_tokens += response.input_tokens
        self._cumulative_output_tokens += response.output_tokens
        self._episode_input_tokens += response.input_tokens
        self._episode_output_tokens += response.output_tokens

        # With SDK tools, actions are queued by game_action/auto_explore tools
        # The action queue is populated before this method is called
        # Use first queued action or "wait" if no action was specified
        has_queue = bool(self._action_queue)
        if self._action_queue:
            action = self._action_queue.pop(0)
            if self._action_queue:
                self._batch_source = "tool_action"
                self._batch_total = len(self._action_queue) + 1
                self._batch_executed = 1
        else:
            action = "wait"  # No action tool was called

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
            reasoning=completion,
            cache_creation_tokens=getattr(response, "cache_creation_tokens", 0) or 0,
            cache_read_tokens=getattr(response, "cache_read_tokens", 0) or 0,
            extended_thinking=getattr(response, "extended_thinking", None),
            action_type="multi" if has_queue else "single",
            compute_requests=None,
            raw_completion=completion,
            ep_llm_calls=self._episode_llm_calls,
            total_llm_calls=self._cumulative_llm_calls,
        )

        return response._replace(reasoning=completion, completion=action)

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

    def _should_abort_queue(self, obs: dict[str, Any]) -> str | None:
        """Check if action queue should be aborted based on observation.

        Returns abort reason string if should abort, None otherwise.
        """
        if not self._action_queue:
            return None

        # Safety limit: abort if explore_room/corridor exceeds max actions
        if self._batch_source in ("explore_room", "explore_corridor"):
            if self._batch_executed >= self._max_explore_actions:
                self.storage.log_error(
                    self.episode_number, self._step,
                    f"Explore limit reached: {self._batch_executed} actions"
                )
                return f"Explore limit reached ({self._batch_executed} actions)"

        text = obs.get("text", {}).get("long_term_context", "")
        text_lower = text.lower()

        # Abort on interactive prompts
        if self._is_interactive_prompt(text):
            return "Interactive prompt detected"

        # Abort on WORSENING hunger (not pre-existing)
        # Only abort if hunger state increased since queue started
        current_hunger = self._extract_hunger(obs)
        if current_hunger is not None and self._queue_start_hunger is not None:
            if current_hunger > self._queue_start_hunger and current_hunger >= 2:
                # Hunger worsened to Hungry (2) or worse
                hunger_names = ["Satiated", "OK", "Hungry", "Weak", "Fainting", "Fainted", "Starved"]
                hunger_name = hunger_names[current_hunger] if current_hunger < len(hunger_names) else f"?{current_hunger}"
                return f"Hunger worsened to {hunger_name}"

        # Pet interactions are normal, don't abort
        pet_patterns = ["swap places", "your kitten", "your cat", "your dog", "your pony",
                        "moves out of your way", "gets out of your way"]
        is_pet_interaction = any(p in text_lower for p in pet_patterns)

        # Abort on combat indicators (but not pet-related)
        combat_patterns = ["hits", "misses", "bites", "attacks", "throws", "swings"]
        if any(p in text_lower for p in combat_patterns) and not is_pet_interaction:
            return "Combat detected"

        # Abort on significant HP drop (>20% since queue started)
        if self._queue_start_hp:
            current_hp = self._extract_hp(obs)
            if current_hp is not None and current_hp < self._queue_start_hp * 0.8:
                return "HP dropped significantly"

        # Abort on danger indicators
        danger_patterns = ["cursed", "poisoned", "confused", "blind", "stuck", "paralyzed"]
        if any(p in text_lower for p in danger_patterns):
            return "Status effect detected"

        # Abort on trap trigger (but not just seeing "trap" in text)
        trap_triggers = ["trigger a", "fall into", "step on a", "you are caught"]
        if any(p in text_lower for p in trap_triggers):
            return "Trap triggered"

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
                        tiles_count = int(np.sum(newly_revealed))
                        self.storage.log_error(
                            self.episode_number, self._step,
                            f"Cautious abort: {tiles_count} tiles revealed nearby"
                        )
                        return f"New tiles discovered nearby ({tiles_count} tiles)"
                # Update snapshot
                self._last_glyphs = glyphs.copy()

        return None

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
        dungeon_num = pos_info[2]
        dlvl = pos_info[3]

        # Get visited tiles for this level
        visited = self.storage.get_visited_for_level(self.episode_number, dungeon_num, dlvl)

        # Re-plan based on exploration type (only room supported, corridor uses far commands)
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

    def _extract_hunger(self, obs: dict[str, Any]) -> int | None:
        """Extract hunger state from observation.

        Returns hunger index: 0=Satiated, 1=OK, 2=Hungry, 3=Weak, 4=Fainting, etc.
        Higher values = more hungry/dangerous.
        """
        raw_obs = obs.get("obs")
        if isinstance(raw_obs, dict):
            blstats = raw_obs.get("blstats")
            if blstats is not None:
                try:
                    # blstats[21] is hunger state in NLE
                    return int(blstats[21])
                except (IndexError, TypeError, ValueError):
                    pass
        return None

    def _extract_position_info(
        self, obs: dict[str, Any]
    ) -> tuple[int, int, int, int] | None:
        """Extract (x, y, dungeon_num, dlvl) from observation blstats."""
        raw_obs = obs.get("obs")
        if isinstance(raw_obs, dict):
            blstats = raw_obs.get("blstats")
            if blstats is not None:
                try:
                    x = int(blstats[0])
                    y = int(blstats[1])
                    dlvl = int(blstats[12])
                    dungeon_num = int(blstats[23])
                    return (x, y, dungeon_num, dlvl)
                except (IndexError, TypeError, ValueError):
                    pass
        return None

    def _format_queue_summary(
        self, aborted: bool = False, reason: str | None = None, remaining: int = 0
    ) -> str:
        """Format summary of queue execution including accumulated messages."""
        executed = self._batch_executed
        total = self._batch_total
        messages = self._queue_messages

        if aborted:
            header = f"[QUEUE ABORTED: {reason}. {remaining} actions cancelled after {executed}/{total} executed.]"
        else:
            header = f"[Queue completed: {executed} actions executed]"

        if not messages:
            return header

        # Deduplicate consecutive identical messages
        unique_msgs: list[str] = []
        for msg in messages:
            if not unique_msgs or msg != unique_msgs[-1]:
                unique_msgs.append(msg)

        # Limit to last 10 messages to avoid overwhelming context
        if len(unique_msgs) > 10:
            unique_msgs = unique_msgs[-10:]
            msg_list = "...\n" + "\n".join(f"- {m}" for m in unique_msgs)
        else:
            msg_list = "\n".join(f"- {m}" for m in unique_msgs)

        return f"{header}\nMessages from NetHack observed during queue execution:\n{msg_list}"

    def _clear_batch_state(self) -> None:
        """Clear all batch/queue tracking state."""
        self._action_queue.clear()
        self._queue_start_hp = None
        self._queue_start_hunger = None
        self._cautious_mode = False
        self._last_glyphs = None
        self._batch_source = None
        self._batch_total = 0
        self._batch_executed = 0
        self._queue_messages.clear()

    def _pop_queued_action(self) -> LLMResponse:
        """Return next queued action without LLM call."""
        action = self._action_queue.pop(0)
        remaining = len(self._action_queue)
        self._batch_executed += 1

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

        # Don't continue if we've hit the safety limit
        if self._batch_executed >= self._max_explore_actions:
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
            detect_room,
            get_position,
            plan_room_exploration,
        )

        pos = get_position(blstats)
        dungeon_num = int(blstats[23])
        dlvl = int(blstats[12])
        visited = self.storage.get_visited_for_level(self.episode_number, dungeon_num, dlvl)

        new_actions: list[str] = []

        # Only room exploration is auto-continued; corridor uses far commands
        if self._batch_source == "explore_room":
            room = detect_room(glyphs, pos)
            if room:
                new_actions = plan_room_exploration(glyphs, room, pos, visited=visited)

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

    def reset(self) -> None:
        """Reset for new episode."""
        super().reset()
        self.episode_number += 1
        self._step = 0
        self._episode_input_tokens = 0
        self._episode_output_tokens = 0
        self._episode_llm_calls = 0
        self._action_queue.clear()
        self._queue_start_hp = None
        self._queue_start_hunger = None
        self._cautious_mode = False
        self._last_glyphs = None
        self._batch_source = None
        self._batch_total = 0
        self._batch_executed = 0
        self._last_dungeon_num = None  # Reset level tracking
        self._last_dlvl = None
        self.storage.log_reset(self.episode_number, str(self.config.client.model_id))

    def on_episode_end(self) -> None:
        """Called when episode ends - close SDK session if active.

        This allows ClaudeToolWrapper to close its session and start fresh
        for the next episode, maintaining clean context boundaries.
        """
        if hasattr(self.client, "close_session"):
            self.client.close_session()
