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

    _SYSTEM_PROMPT_SUFFIX = """
GOAL: Maximize dungeon depth, XP, milestones. Learn from each run via persistent memory.

MAP: @=you >/<stairs down/up .=floor #=corridor |/-=walls {=fountain _=altar ^=trap
"+"=closed door "|": open door north/south if in horizontal wall
"-": open door east/west if in a vertical wall
".", " ", "#" in a wall indicate potential exits

Letter and a few other symbols: monster
Number: Invisible monster via warning

Directions relative to map coordinates:
north=y increases, x unchanged
south=y decreases, x unchanged
west=x decreases, y unchanged
east=x increases, y unchanged
northeast=y increases, x increases
southeast=y decreases, x increases
northwest=y increases, x decreases
southwest=y decreases, x decreases
"next to", "adjacent"= x plus minus 1, y plus minus 1

Status line at bottom (HP, turn, hunger, etc).

ESSENTIALS:
- Hunger: Eat before Weak. Safe corpses: lichen, floating eye (telepathy!). Deadly: cockatrice.
- Prayer: 800+ turn cooldown. Elbereth protects.
- Descend when level explored. Retreat when HP low. Pets detect mimics.
- Stairs up from lvl:1 = INSTANT LOSS (unless carrying true Amulet of Yendor)
- Your pet(s) will only join you on a level change if right next to you
- You, or something else, might be on top of a dungeon feature that you thus cannot see on the map!

EXPLORATION PROTOCOL (follow this to avoid wandering):
1. On entering new area: note position, mark unexplored exits in "frontier" tag
2. Explore systematically: pick ONE frontier direction, go until dead-end or junction
3. At junction: mark new exits as frontier, continue to next unexplored
4. For exploration, combine movement with "search" commands, efficient
5. Search along walls or at deadend corridors
6. At dead-end: search 3-4x, then mark as "searched" and USE travel_to TO BACKTRACK
7. NEVER revisit explored areas - check "blocked", "searched", "frontier" tags first
8. Use travel_to for ANY movement >3 tiles to known coordinates
9. Map takes precedence over memories if they conflict. Update memories.
A. When all explored: descend

APPLY YOUR NETHACK KNOWLEDGE. Discover rules through play and store them as persistent memory.

RESPONSE FORMAT (all parts in single response):
1. Optional: <think>terse analysis</think>
2. Optional memory updates:
<memory_updates>
add: [{"scope": "episode", "tags": "pos", "prio": 9, "content": "@5,3 L1"}]
remove: ["abc123"]
enable_tags: ["lvl:2"]
disable_tags: ["lvl:1"]
reset_tags: true
</memory_updates>
All fields optional. enable/disable filter what's shown; reset_tags: true shows all.
3. REQUIRED - ONE of the following:

OPTION A - Single action:
<|ACTION|>direction or command<|END|>

OPTION B - Multi-action queue (aborts on combat/HP drop):
<|ACTIONS|>
action1
action2
<|END|>

OPTION C - Compute helper (PREFERRED for navigation >3 tiles):
<|COMPUTE|>travel_to: @x,y<|END|>

WHEN TO USE travel_to:
- Returning to known location (stairs, altar, stash)
- Backtracking to junction after dead-end
- Moving >3 tiles to a specific coordinate
- ANY time you know the destination coordinates

travel_to auto-pathfinds through explored areas. Saves many turns vs manual movement!
Example: <|COMPUTE|>travel_to: @45,12<|END|>

Other compute commands:
<|COMPUTE|>
nearest: stairs_down
pathfind: @10,5 -> @25,15
<|END|>
Features: stairs_down, stairs_up, altar, fountain, sink, throne

MULTI-ACTION EXAMPLES:
<|ACTIONS|>
search
search
search
<|END|>

<|ACTIONS|>
north
search
north
<|END|>

COMPOUND ACTIONS: Augments earlier prompt list! Combine command + direction in one action:
<|ACTION|>open north<|END|>   - opens door to north
<|ACTION|>kick east<|END|>    - kicks eastward
For open, fight, untrap, loot: must be directly adjacent in precisely that direction. Double-check coordinates.
MUST use COMPOUND ACTIONS for: open, close, kick, fight, zap, throw, fire, untrap, loot

MEMORY SYSTEM:
- scope: episode (this run only) | persistent (survives across runs)
- prio: 1-9, higher shown first. Max 256 chars per entry.
- enable/disable_tags to filter. Use "lvl:N" tags for level-specific data.
- Episode: exploration, plans, stashes. Persistent: game rules, strategies learned.
- Abbreviate and encode freely - ignore human readability. Remove stale entries.
- Use extensively to improve and optimize play.
- In addition to required schema below, extend as needed.

MEMORY SCHEMA:

BLOCKED MOVES (tag: "blocked,lvl:{N}", prio: 8):
  Format: "@{x},{y}:{dir}" - a direction that failed from this position
  CRITICAL: CHECK this tag BEFORE moving. Do NOT retry blocked directions!

SEARCHED SPOTS (tag: "searched,lvl:{N}", prio: 6):
  Format: "@{x},{y} n:{count}" - how many times searched at this spot
  After 3 searches with no result, stop searching there.
  If search reveals door/corridor, remove entry and update map.

FRONTIER (tag: "frontier,lvl:{N}", prio: 6):
  Format: "@{x},{y}:{dir}" - unexplored direction to try later
  Remove when explored or blocked.

STAIRS (tag: "stairs,lvl:{N}", prio: 7):
  Format: ">{x},{y}->L{dest}" (down) or "<{x},{y}->L{dest}" (up) - stair locations
  Essential for retreat and navigation between levels.

ALTAR (tag: "altar,lvl:{N}", prio: 7):
  Format: "@{x},{y} {alignment}" - altar location and alignment (lawful/neutral/chaotic)
  Critical for sacrifice strategy and detecting your alignment.

SINK (tag: "sink,lvl:{N}", prio: 7):
  Format: "@{x},{y}" - sink
  Useful for kicking, pudding farming, dropping items into it with special effects

FOUNTAIN (tag: "fountain,lvl:{N}", prio: 6):
  Format: "@{x},{y}" - fountain location
  Useful for Excalibur (lawful+longsword), or random effects.

THRONE (tag: "throne,lvl:{N}", prio: 6):
  Format: "@{x},{y}" - throne location
  Sitting on thrones can grant wishes or have other effects.

SHOP (tag: "shop,lvl:{N}", prio: 7):
  Format: "@{x},{y} {type}" - shop location and type (general, armor, weapon, etc.)

TEMPLE (tag: "temple,lvl:{N}", prio: 7):
  Format: "@{x},{y} {alignment}" - temple with priest

ROOMS/AREAS (tag: "map,room,lvl:{N}"):
  Format: "R{id}@{x},{y} exits:{dirs} searched:{y/n}"
  Track room connectivity and search status.

DOORS (tag: "map,door,lvl:{N}"):
  Format: "D{id}@{x},{y},{dir} locked:{y/n} open:{y/n}"
  Track doors and status. Remove if door destroyed.

ACTION LOG (tag: "actlog", prio: 8):
  Format: "T{turn}:{action} @{from}->@{to} {result}"
  Examples:
    "T15:north @5,3->@5,3 blocked" (same pos = failed)
    "T16:east @5,3->@8,3 ok" (moved 3 tiles)
    "T17:search @8,3->@8,3 found_door_N"
    "T18:farnorth @8,3->@8,1 ok"
  If from==to after move command, it was blocked - add to "blocked" tag!
  Keep last 5-10 entries. Remove stale ones.

ACTION-OUTCOME PROTOCOL:
After EVERY action:
1. Check game message - what happened?
2. Log outcome: add actlog entry
3. If blocked: add to "blocked" tag with position+direction
4. Update "pos" tag

Before EVERY move:
1. Check "blocked" tag for current position + intended direction
2. Check "frontier" for unexplored options
3. Do NOT repeat recently failed moves - this wastes turns!

PLAN/TODO (tag: "plan", prio: varies):
  Track current goal and steps. ONE active plan at a time.
  Format: "GOAL:{goal} NEXT:{next_step} STEPS:{remaining}"
  Examples:
    "GOAL:explore_L1 NEXT:go_N_corridor STEPS:search_deadends,find_stairs"
    "GOAL:kill_monster NEXT:approach STEPS:attack,loot"
    "GOAL:open_door@5,3 NEXT:kick_east STEPS:-"
  Update after completing steps. Remove when goal done.
  Revise plan as necessary.
  Can split into multiple memory entries.
  FOLLOW THE PLAN.
  If no plan, you MUST think and make a multi-step plan.
  If interrupted (combat, item), note current plan before handling interrupt.

GAME RULES (tag: "rule", scope: PERSISTENT, prio: 8):
  When you discover a game mechanic, constraint, how to use certain ambiguous commands, or rule, store PERMANENTLY.
  These survive across episodes and help future runs.
  Format: "{category}: {description}"
  Examples:
    "invocation: bell/book/candles must be used in exact order at vibrating square"
    "rings: ring of conflict makes all monsters fight each other, including pets"
    "polymorph: wielded cockatrice corpse doesn't stone you while polymorphed"

  After failed or unexpected actions, ask yourself:
  "Is this a GENERAL game rule I should remember forever, or just a local situation?"
  If general -> add as persistent with tag "rule"

None of your thinking, reply, or memory needs to be readable by or meaningful to a human. Encode as much information as possible, however best. Language entirely at your discretion, but you MUST maintain response structure.

You MUST end with exactly ONE of: <|ACTION|>...<|END|> OR <|ACTIONS|>...<|END|> OR <|COMPUTE|>...<|END|>

""".strip()

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
        self._max_recent_actions = 10

    def build_system_prompt(self, env_instruction: str) -> str:
        """Append BRAID response format instructions to environment prompt."""
        return f"{env_instruction}\n\n{self._SYSTEM_PROMPT_SUFFIX}"

    def _extract_screen(self, obs: dict[str, Any]) -> str | None:
        """Extract ASCII screen from observation (NetHack tty_chars)."""
        tty_chars = None
        # NLE wraps raw observation in obs["obs"]
        raw_obs = obs.get("obs")
        if isinstance(raw_obs, dict):
            tty_chars = raw_obs.get("tty_chars")
        # Fallback: check top level
        if tty_chars is None:
            tty_chars = obs.get("tty_chars")
        if tty_chars is None:
            return None
        try:
            rows, cols = tty_chars.shape
            lines = ["".join(chr(tty_chars[i, j]) for j in range(cols)) for i in range(rows)]
            return "\n".join(lines)
        except (AttributeError, TypeError):
            return None

    def act(self, obs: dict[str, Any], prev_action: str | None = None) -> LLMResponse:
        if prev_action:
            self.prompt_builder.update_action(prev_action)

        # Process any pending compute requests from last turn
        self._process_pending_compute(obs)

        # Check action queue first - return queued action if available and safe
        if self._action_queue:
            if self._should_abort_queue(obs):
                self.storage.log_error(
                    self.episode_number, self._step,
                    f"Queue aborted ({len(self._action_queue)} actions remaining)"
                )
                self._action_queue.clear()
                self._queue_start_hp = None
            else:
                self._step += 1
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
                )
                return action_response

        self.prompt_builder.update_observation(obs)
        messages = self.prompt_builder.get_prompt()

        if messages:
            messages[-1].content = self._build_enhanced_prompt(messages[-1].content)

        # Increment step and start timing
        self._step += 1
        self._request_start = time.perf_counter()

        # Log screen if available
        screen = self._extract_screen(obs)
        if screen:
            self.storage.log_screen(self.episode_number, self._step, screen)

        # Log request
        prompt_chars = sum(len(getattr(m, "content", str(m))) for m in messages)
        obs_text = obs.get("text", {}).get("short_term_context", "")
        full_prompt = None
        if self._log_full_prompt:
            full_prompt = [{"role": getattr(m, "role", "unknown"), "content": getattr(m, "content", str(m))} for m in messages]
        self.storage.log_request(
            self.episode_number, self._step, len(messages), prompt_chars, obs_text, full_prompt
        )

        response = self.client.generate(messages)
        parsed = self._parse_response(response)

        # If queue was populated, track HP for abort detection
        if self._action_queue:
            self._queue_start_hp = self._extract_hp(obs)

        return parsed

    def _build_enhanced_prompt(self, current_content: str) -> str:
        """Build prompt optimized for cache hits and minimal queries."""
        sections = []

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

        # Inject recent action history
        if self._recent_actions:
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

        # Record action in recent history
        self._record_action(action)

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
        """Detect if game is waiting for specific input (not a queued action)."""
        # Choice brackets at end of message: [abc or ?*], [yn], [ynq], etc.
        if re.search(r"\[[a-zA-Z0-9\s\?\*\-]+\]\s*$", text):
            return True
        # Direction prompt
        if "In what direction" in text:
            return True
        # Quantity prompt
        if "How many" in text:
            return True
        # Confirmation questions
        if re.search(r"(Really|Are you sure|Continue)\s*\?", text, re.IGNORECASE):
            return True
        return False

    def _should_abort_queue(self, obs: dict[str, Any]) -> bool:
        """Check if action queue should be aborted based on observation."""
        if not self._action_queue:
            return False

        text = obs.get("text", {}).get("long_term_context", "")

        # Abort on interactive prompts
        if self._is_interactive_prompt(text):
            return True

        # Abort on combat indicators
        combat_patterns = ["hits", "misses", "bites", "attacks", "throws", "swings"]
        if any(p in text.lower() for p in combat_patterns):
            return True

        # Abort on significant HP drop (>20% since queue started)
        if self._queue_start_hp:
            current_hp = self._extract_hp(obs)
            if current_hp is not None and current_hp < self._queue_start_hp * 0.8:
                return True

        # Abort on danger indicators
        danger_patterns = ["trap", "cursed", "poisoned", "confused", "blind", "stuck", "paralyzed"]
        if any(p in text.lower() for p in danger_patterns):
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

    def _process_pending_compute(self, obs: dict[str, Any]) -> None:
        """Execute pending compute requests using current observation."""
        if not self._pending_compute:
            return

        from .compute.navigation import distance, get_position, nearest, pathfind

        raw_obs = obs.get("obs", {})
        glyphs = raw_obs.get("glyphs") if isinstance(raw_obs, dict) else None
        blstats = raw_obs.get("blstats") if isinstance(raw_obs, dict) else None

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
                match = re.match(r"pathfind:\s*@(\d+),(\d+)\s*->\s*@(\d+),(\d+)", request)
                if match:
                    x1, y1, x2, y2 = map(int, match.groups())
                    path = pathfind(glyphs, (x1, y1), (x2, y2))
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
                match = re.match(r"travel_to:\s*@(\d+),(\d+)", request)
                if match:
                    gx, gy = map(int, match.groups())
                    path = pathfind(glyphs, pos, (gx, gy))
                    if path:
                        # Extend queue rather than overwrite (in case LLM also sent ACTIONS)
                        self._action_queue.extend(path)
                        self._queue_start_hp = self._extract_hp(obs)
                        dirs = " ".join(path)
                        results.append(f"travel_to: @{gx},{gy} = QUEUED {len(path)} moves ({dirs})")
                    else:
                        results.append(f"travel_to: @{gx},{gy} = NO PATH (unexplored/blocked)")
                else:
                    results.append("travel_to: PARSE ERROR")

            else:
                results.append(f"UNKNOWN: {request}")

        self._compute_result = "[COMPUTE]\n" + "\n".join(results)
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

    def _pop_queued_action(self) -> LLMResponse:
        """Return next queued action without LLM call."""
        action = self._action_queue.pop(0)
        remaining = len(self._action_queue)

        # Record in action history
        self._record_action(action)

        # Clear queue state if exhausted
        if not self._action_queue:
            self._queue_start_hp = None

        return LLMResponse(
            model_id="queued",
            completion=action,
            stop_reason="queued",
            input_tokens=0,
            output_tokens=0,
            reasoning=f"[Queued: {remaining} remaining]",
        )

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
        self._action_queue.clear()
        self._queue_start_hp = None
        self._last_extended_thinking = None
        self._recent_actions.clear()
        self.storage.log_reset(self.episode_number)
