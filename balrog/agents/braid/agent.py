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

    _SYSTEM_PROMPT_SUFFIX = """
Your long-term goal beyond this single playthrough is to get better at playing NetHack.

You are scored on achieving certain milestones in the game, character level, dungeon depth, eventual ascension. You should strive to achieve those goals in an optimal number of turns without dying.

ADDITIONAL INFORMATION BEFORE OBSERVATIONS:

SOME MAP SYMBOLS:
@=you >/<stairs .=floor #=corridor |/-=walls +=closed door '=open door
{=fountain _=altar ^=trap Letters=monsters (lowercase weaker)
0=boulder (push with direction) `=rock/statue $=gold )=weapon [=armor
!=potion ?=scroll /=wand ==ring "=amulet (=tool *= gem/stone

On the ASCII map: North is up. South is down. East is right. West is left.

Character status (values, alignment, current XP and to next level, etc, Turn number) is provided at the bottom of the map.

INCOMPLETE MONSTER THREATS BY LETTER:
a=ant(giant) b=blob c=cockatrice(DEADLY touch!) d=dog/jackal e=eye(floating paralyze!)
f=cat g=gremlin h=humanoid i=imp j=jelly k=kobold l=leprechaun(steals gold)
m=mimic n=nymph(steals items!) o=orc p=piercer q=quadruped r=rodent s=spider
t=trapper u=unicorn v=vortex w=worm x=xan/grid bug y=light z=zruty
A=Angel B=Bat C=Centaur D=Dragon(VERY dangerous) E=Elemental F=Fungus/lichen
G=Gnome H=Giant I=Invisible stalker J=Jabberwock K=Keystone Kop L=Lich(undead spellcaster)
M=Mummy N=Naga O=Ogre P=Pudding Q=Quantum mechanic R=Rust monster(destroys armor!)
S=Snake T=Troll(regenerates) U=Umber hulk V=Vampire W=Wraith X=Xorn Y=Yeti Z=Zombie
@=human(shopkeeper,guard,priest,nurse) &=demon '=golem ;=sea monster :=lizard/newt

SOME SAFE CORPSES: lichen, newt, jackal, kobold, orc, gnome, dwarf, floating eye(telepathy!), giants
RISKY: tengu (can grant teleport control, poison resistance, but also teleportitis)
DANGEROUS CORPSES: cockatrice(instant death), Medusa (stoning)
NEVER EAT: anything you're unsure about. When in doubt, don't eat it. Could be rotten. Tinning helps.

KEY STATUS EFFECTS:
Hungry->Weak->Fainting->death. Eat before Weak. Conf=confused, Stun=stunned, Blind, Hallu=hallucinating
Ill=sick(pray or cure), FoodPois=food poisoning(pray). Burdened/Stressed=carrying too much, Stoning=Eating lizard corpose, acidic monster, potion of acid, stone to flesh spell, successful prayer

DUNGEON BRANCHES:
Dungeons of Doom (main): levels 1-~26, goal is to descend
Gnomish Mines: entrance ~lvl 2-4, has Minetown with shops, temples, perhaps better gear
Rogue level: "You enter what seems to ben an older, more primitive world.", lvl 15-18. Slightly different, more archaic rules and symbols.
Sokoban: entrance ~lvl 5-9, puzzle branch with guaranteed useful items at end, can't move diagonally, boulders need to be pushed into pits, solve carefully with a plan
Oracle: ~lvl 5-9, can consult for sometimes useful tips (costs gold)
Castle: ~lvl 25, has wand of wishing in chest
Gehennom: below Castle, fire and demons, working toward Amulet, many special levels

KEY STRATEGIES:
- Elbereth: engrave in dust (E then write "Elbereth", possibly using a wand of fire etc). Most monsters won't attack you on it. Moving might harm it. Safe spot for stashes.
- Altar sacrifice: kill monsters, offer corpses on aligned altar for favor and gifts (artifact weapons!), risky at non-aligned altars
- Price ID: in shops, base prices can reveal item identity (e.g., 300zm scroll = identify)
- Priests: Giving them gold (between 200 to 400 times player level) can grant intrinsic protection
- Wand testing: engrave letter with wand, message can reveal wand type
- When in an apparent deadend, search a few times before marking as such (you can queue up a multi-action)
- Search might also reveal secret doors in walls
- Certain rings and character abilities and eating (tinned) corpses can convey auto-search
- Stealth is very important to acquire
- Fountain: quaff for random effects, dip for Excalibur if lawful with long sword
- Stash: leave items on early levels to retrieve later
- Pet: keep fed (throw food), can steal from shops, detects mimics/traps
- Remember mimics exist
- Monsters could be invisible
- Tinning has beneficial effects
- Magic markers should be blessed, have or create blessed scrolls of charging
- Explore and move efficiently, do not waste movements, choose a consistent pathing pattern
- The "far" actions allow you to move in a certain direction as far as possible, until an obstacle appears, and are very efficient for exploration (preferable to multi actions, even)
- Remember which areas you have searched, or tried to explore, so you do not need to repeat
- You can often move boulders, unless they're up against a dead end (wall, another boulder, or monster)
- You can only unlock, open, or kick doors when you are right next to them, and the next prompt from the game will then be which direction, so make a note for the next turn to answer this fast
- Remember your existing NetHack knowledge

EARLY GAME PRIORITIES:
1. Find and equip any armor/weapons
2. Identify food sources, stockpile rations
3. Find altar for sacrifice and alignment
4. Don't over-explore - descend when level explored
5. Keep track of stairs up for retreat
6. Improve player level, but don't outgrow your equipment / DPS, as monsters grow with your level
7. Make a plan (via memories), aware of LLM limitations in this BALROG NetHack Learning Environment, so you can make consistent progress

COMMON MISTAKES TO AVOID:
- Fighting when low HP instead of retreating
- Eating unknown corpses (especially cockatrice!)
- Most rings increase hunger (except slow digestion, very desirable!)
- Invoking spells increases hunger
- Praying too often (gods get angry, wait 800+ turns)
- Attacking shopkeeper/temple priest (very dangerous)
- Ignoring hunger until Fainting
- Fighting multiple too strong enemies in open spaces
- Forgetting where stairs up are located
- Using unidentified wands pointed at self
- Stepping on traps repeatedly (use search to find them)
- You can only move far in a direction if at least the first step is possible
- You CANNOT move into walls, doors, or the surrounding walls. This is NOT an effective exploration strategy. Use search instead, and mark the results in your memory.

RESPONSE FORMAT (all parts in single response):
1. Optional thinking if situation requires it according to your judgment: <think>terse analysis</think>
2. Optional memory updates to optimize future actions:
<memory_updates>
add: [{"scope": "episode|persistent", "tags": "t1,t2", "prio": 5, "content": "..."}]
remove: ["entry_id"]
enable_tags: ["tag"] | disable_tags: ["tag"] | reset_tags: true
</memory_updates>
3. REQUIRED action - single action from system prompt OR multi-action sequence:
   Single: <|ACTION|>your_action<|END|>
   Multi:  <|ACTIONS|>
           action1
           action2
           action3
           <|END|>

MULTI-ACTION GUIDELINES:
- Allows for more efficient and cost-effective playthroughs. Prioritize use!
- Use for: navigation sequences, repeated search, search + navigation combos, safe paths
- You can issue multiple move commands even in unknown paths. If it's a deadend, you'll simply stop.
- Avoid for: combat, unknown areas, low HP
- Queue aborts automatically on: combat, prompts requiring response, HP drop, traps
- You can queue as many multi-actions as needed

MEMORY:
- The memory tool is your best chance for making progress and avoiding getting stuck. Use extensively, and very detailed.
- scope: episode (cleared each play-through) | persistent (survives)
- T: indicates the step/turn at which this observation was recorded. (Compare to T: on map)
- prio: 1-9, higher shown first when limit reached (default 5)
- enable/disable_tags: filter what's included in prompt; reset_tags: true to include all
- update/replace: indirectly, map to remove + add
- You're also provided with a list of existing tags and how many entries have them

HINTS FOR MEMORY USE:
- content/tags exclusive for agent only - abbreviate freely, disregard human readability, encode as much information as possible, regardless of language used
- Use tags for specific levels, areas, monsters, puzzles, short- and long-term planning, risk tracking, specific to character role, ...
- Use episode memory for tracking exploration, stashes, plans, etc: anything that is only for this particular playthrough attempt
- In particular detail: track areas, corridors, rooms, levels you have already explored, directions you have already tried to move in but made no progress, and directions/areas you plan on exploring in the future, to avoid getting stuck in loops
- Use it to annotate the explored map, corridors, rooms, levels
- Create enough memories to improve your progress in future turns with minimal repetition
- Use memory as a todo list and planner
- Use persistent memory to learn permanently and across runs, properties of monsters items etc, both tactically and strategically or meta attributes such as establishing a coherent tagging strategy for memory, supplementing and overriding system prompt hints for play
- When you discover limitations, constraints, or invalid moves, create memories so you can take this into account in the future. When those are fundamental rules of the game, store as persistent memory.
- A single entry can have upto 256 characters. Split if needed.
- You can have hundreds of entries with proper filtering, showing up to a 100 for persistent and episode memories each
- Remove entries that truly no longer apply

None of your thinking, reply, or memory needs to be readable by or meaningful to a human. Encode as much information as possible, however best. Language entirely at your discretion, but you MUST maintain response structure.

At each turn, you are provided with this prompt, the previous observations and your past actions, the memory entries for enabled tags, and the current observation (map screenshot).

Note that the language observation is an incomplete rendering of the map meant to augment weaker LLMs. Actual ASCII map takes precedence.

You MUST end with a valid single ACTION or multi ACTIONS sequence.

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
            header = f"PERSISTENT ({len(p_entries)}"
            if len(p_entries) >= self.max_memory_context:
                total = self.storage.count(tags=self._enabled_tags, scope=MemoryScope.PERSISTENT)
                hidden = total - len(p_entries)
                if hidden > 0:
                    header += f"+{hidden} hidden due to limit"
            header += "):"
            sections.append(f"{header}\n" + "\n".join(lines))

        if e_entries:
            lines = [f"[{e.entry_id}] @{e.source_step or '?'} (prio:{e.priority}) (tags: {e.tags}) {e.content}" for e in e_entries]
            header = f"EPISODE ({len(e_entries)}"
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
        )

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

    def _parse_multi_actions(self, completion: str) -> list[str]:
        """Parse single action or multi-action block from completion."""
        # Check for <|ACTIONS|>...<|END|> block (multi-action)
        multi_match = re.search(r"<\|ACTIONS\|>(.*?)<\|END\|>", completion, re.DOTALL)
        if multi_match:
            actions = [a.strip() for a in multi_match.group(1).strip().split("\n") if a.strip()]
            return actions if actions else [self._fallback_action(completion)]

        # Check for single <|ACTION|>...<|END|> (normal case)
        single_match = re.search(r"<\|ACTION\|>(.*?)<\|END\|>", completion, re.DOTALL)
        if single_match:
            return [single_match.group(1).strip()]

        # Fallback
        return [self._fallback_action(completion)]

    def _pop_queued_action(self) -> LLMResponse:
        """Return next queued action without LLM call."""
        action = self._action_queue.pop(0)
        remaining = len(self._action_queue)

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
        self.storage.log_reset(self.episode_number)
