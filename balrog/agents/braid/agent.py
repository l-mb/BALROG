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
    - Label filtering: LLM can enable/disable labels to control what's shown
    - Adaptive thinking: LLM self-selects depth of reasoning per step
    """

    _SYSTEM_PROMPT_SUFFIX = """
ADDITIONAL INFORMATION BEFORE OBSERVATIONS:

SOME MAP SYMBOLS:
@=you >/<stairs .=floor #=corridor |/-=walls +=closed door '=open door
{=fountain _=altar ^=trap Letters=monsters (lowercase weaker)
0=boulder (push with direction) `=rock/statue $=gold )=weapon [=armor
!=potion ?=scroll /=wand ==ring "=amulet (=tool *= gem/stone

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
Sokoban: entrance ~lvl 5-9, puzzle branch with guaranteed useful items at end, can't move diagonally, solve carefully
Oracle: ~lvl 5-9, can consult for sometimes useful tips (costs gold)
Castle: ~lvl 25, has wand of wishing in chest
Gehennom: below Castle, fire and demons, working toward Amulet

KEY STRATEGIES:
- Elbereth: engrave in dust (E then write "Elbereth"). Most monsters won't attack you on it.
- Altar sacrifice: kill monsters, offer corpses on aligned altar for favor and gifts (artifact weapons!), risky at non-aligned altars
- Price ID: in shops, base prices can reveal item identity (e.g., 300zm scroll = identify)
- Priests: Giving them gold (between 200 to 400 times player level) can grant intrinsic protection
- Wand testing: engrave letter with wand, message can reveal wand type
- Fountain: quaff for random effects, dip for Excalibur if lawful with long sword
- Stash: leave items on early levels to retrieve later
- Pet: keep fed (throw food), can steal from shops, detects mimics/traps
- Remember mimics exist
- Monsters could be invisible
- Tinning has beneficial effects
- Magic markers should be blessed, have or create blessed scrolls of charging
- Explore efficiently, do not waste movements

EARLY GAME PRIORITIES:
1. Find and equip any armor/weapons
2. Identify food sources, stockpile rations
3. Find altar for sacrifice and alignment
4. Don't over-explore - descend when level cleared
5. Keep track of stairs up for retreat
6. Improve player level, but don't outgrow your equipment / DPS, as monsters grow with your level

COMMON MISTAKES TO AVOID:
- Fighting when low HP instead of retreating
- Eating unknown corpses (especially cockatrice!)
- Most rings increase hunger
- Praying too often (gods get angry, wait 800+ turns)
- Attacking shopkeeper/temple priest (very dangerous)
- Ignoring hunger until Fainting
- Fighting multiple too strong enemies in open spaces
- Forgetting where stairs up are located
- Using unidentified wands pointed at self
- Stepping on traps repeatedly (use search)

RESPONSE FORMAT:
1. Assess: THINKING_NEEDED: yes/no
2. If yes: <think>your terse analysis, maximize signal to noise, optimize for your own use</think>
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

HINTS FOR MEMORY USE:
- content/tags exclusive for agent only - abbreviate freely, disregard human readability, encode as much information as possible, regardless of language used
- Could use labels for specific levels, areas, monsters, puzzles, short- and long-term planning, risk tracking, specific to character role, ...
- Use persistent memory to learn permanently and across runs, both tactically and strategically or meta attributes such as labelling strategy for memory, supplementing and overriding system prompt hints for play
- Your long-term goal beyond this one episode is to get good at playing NetHack!

The following includes several observations, your past actions, and the latest observation.
Note that the language observation is an incomplete rendering of the map.

Memories are provided to you later.

""".strip()

    def __init__(self, client_factory: Any, prompt_builder: Any, config: Any):
        super().__init__(client_factory, prompt_builder)
        self.config = config
        braid_cfg = config.agent.braid

        # Initialize unified storage (SQLite with WAL for multi-worker support)
        self.storage = BraidStorage(Path(braid_cfg.db_path))
        self.max_memory_context = braid_cfg.get("max_persistent_context", 40)
        self.episode_number = self.storage.max_episode()
        self._enabled_labels: set[str] | None = None

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
        self._label_changes = False
        self._added_entries: list[dict[str, Any]] = []
        self._removed_ids: list[str] = []

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
        return self._parse_response(response)

    def _build_enhanced_prompt(self, current_content: str) -> str:
        """Build prompt optimized for cache hits and minimal queries."""
        sections = []

        p_entries = self.storage.retrieve(
            tags=self._enabled_labels, scope=MemoryScope.PERSISTENT, limit=self.max_memory_context
        )
        e_entries = self.storage.retrieve(
            tags=self._enabled_labels, scope=MemoryScope.EPISODE,
            episode=self.episode_number, limit=self.max_memory_context
        )

        if p_entries:
            lines = [f"[{e.entry_id}] (prio:{e.priority}) (tags: {e.tags}) {e.content}" for e in p_entries]
            header = f"PERSISTENT ({len(p_entries)}"
            if len(p_entries) >= self.max_memory_context:
                total = self.storage.count(tags=self._enabled_labels, scope=MemoryScope.PERSISTENT)
                hidden = total - len(p_entries)
                if hidden > 0:
                    header += f"+{hidden} hidden due to limit"
            header += "):"
            sections.append(f"{header}\n" + "\n".join(lines))

        if e_entries:
            lines = [f"[{e.entry_id}] (prio:{e.priority}) (tags: {e.tags}) {e.content}" for e in e_entries]
            header = f"EPISODE ({len(e_entries)}"
            if len(e_entries) >= self.max_memory_context:
                total = self.storage.count(
                    tags=self._enabled_labels, scope=MemoryScope.EPISODE, episode=self.episode_number
                )
                hidden = total - len(e_entries)
                if hidden > 0:
                    header += f"+{hidden} hidden due to limit"
            header += "):"
            sections.append(f"{header}\n" + "\n".join(lines))

        label_info = self._compute_label_summary(p_entries + e_entries)
        if label_info:
            sections.append(label_info)

        sections.append(current_content)
        sections.append(self._get_action_instructions())

        return "\n\n".join(sections)

    def _compute_label_summary(self, entries: list[MemoryEntry]) -> str:
        """Compute label summary from already-retrieved entries."""
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
        )

        # Log memory updates
        self.storage.log_memory_update(
            episode=self.episode_number,
            step=self._step,
            adds=self._mem_adds,
            removes=self._mem_removes,
            label_changes=self._label_changes,
            added_entries=self._added_entries if self._log_memory_details else None,
            removed_ids=self._removed_ids if self._log_memory_details else None,
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

                    entry_id = self.storage.store(
                        MemoryEntry(
                            tags=tags,
                            content=content,
                            scope=scope,
                            priority=prio,
                            source_episode=self.episode_number,
                        )
                    )
                    self._mem_adds += 1
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
                        self.storage.remove(entry_id)
                        self._mem_removes += 1
                        if self._log_memory_details:
                            self._removed_ids.append(entry_id)
            except json.JSONDecodeError:
                self.storage.log_error(self.episode_number, self._step, "Failed to parse memory removals")

        # Parse enable_labels
        enable_match = re.search(r"enable_labels:\s*\[(.+?)\]", mem_text, re.DOTALL)
        if enable_match:
            try:
                labels = json.loads(f"[{enable_match.group(1)}]")
                labels = {str(lbl).lower().strip() for lbl in labels if isinstance(lbl, str)}
                if labels:
                    if self._enabled_labels is not None:
                        self._enabled_labels |= labels
                    self._label_changes = True
            except json.JSONDecodeError:
                self.storage.log_error(self.episode_number, self._step, "Failed to parse enable_labels")

        # Parse disable_labels
        disable_match = re.search(r"disable_labels:\s*\[(.+?)\]", mem_text, re.DOTALL)
        if disable_match:
            try:
                labels = json.loads(f"[{disable_match.group(1)}]")
                labels = {str(lbl).lower().strip() for lbl in labels if isinstance(lbl, str)}
                if labels:
                    if self._enabled_labels is None:
                        all_tags = self.storage.all_tags()
                        self._enabled_labels = all_tags - labels
                    else:
                        self._enabled_labels -= labels
                    self._label_changes = True
            except json.JSONDecodeError:
                self.storage.log_error(self.episode_number, self._step, "Failed to parse disable_labels")

        # Parse reset_labels
        reset_match = re.search(r"reset_labels:\s*(true|false)", mem_text, re.IGNORECASE)
        if reset_match and reset_match.group(1).lower() == "true":
            self._enabled_labels = None
            self._label_changes = True

    def reset(self) -> None:
        """Reset for new episode."""
        super().reset()
        self.episode_number += 1
        self._step = 0
        self._enabled_labels = None
        self._episode_input_tokens = 0
        self._episode_output_tokens = 0
        self.storage.log_reset(self.episode_number)
