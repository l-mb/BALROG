# SUSE HackWeek 2025: BALROG Recurrent Agentic Iterative Dungeoneer (BRAID)

**Author:** Lars Marowsky-Brée (2025-12)

## Introduction

> _The only winning move is not to play._ (WarGames, 1983)

The movie WarGames fascinated me in childhood and set me on the path towards software engineering. The iconic takeaway from the movie is the "artificial intelligence" WOPR realizing the futility of war through playing Tic-Tac-Toe. The first game I ever programmed myself was thus a (very, very simplistic, since I would only learn about game theory many years later) "learning" TTT implementation, on a Sharp MZ-821.

Forward a few years, and I'd encounter NetHack - whose minimal aesthetics, rich lore, and charm let me to spend uncountable hours exploring the procedurally generated dungeons, with significant complexity in available actions and encounters, arcane actions, and unpredictable, pseudo-random playthroughs.

Fast-forward a few decades, and Generative & Agentic AI are touted as capable of taking over even complex tasks requiring long-term planning, running for hours in pursuit of goals, supposedly even in the real world.

## SUSE HackWeek Goals

I had little to no experience with implementing any such solutions myself, and only been exposed to chat bots or coding assistants and agentic coding tools like Gemini-CLI and Claude Code as a user; which - ethics and sustainability questions aside - can achieve some somewhat impressive outcomes, especially in text environments.

I decided to explore this space as a learning exercise as part of my SUSE's HackWeek 2025. Using game environments for assessing and even furthering Machine Learning and/or "Artificial Intelligence" is a well-established approach even beyond science fiction movies; famously, the (deterministic) games of chess and Go (DeepMind).

What better place to start than a pseudo-random terminal game I was very familiar with, and seeing if I could make an agent play instead of me, while I would be free to work? (... humanity has gone wrong somewhere.)

This is not as eclectic as it might appear. Using a complex game tests the Agentic AI's capability to:
- Follow a somewhat complex rule set
- Handle a complex set of possible actions in response to dynamically evolving environments
- Decipher structured inputs
- Correctly reference specific details from its context window
- Use tools that were not immediately part of its training set
- Long session context
- Maintain a short- and long-term target
- Apply the "world knowledge" from its training data
- Allow comparison between different models for the same task

## Progress during the week

A brief search showed I was clearly not the first to have this idea: the NetHack Learning Environment was created by [Facebook research in 2020](https://github.com/facebookresearch/nle/releases/tag/v0.1.0), though it has now moved to a [dedicated GitHub org](https://github.com/NetHack-LE/nle?tab=readme-ov-file)). Effort and research turned into a few academic publications, leading to [BALROG](https://arxiv.org/abs/2411.13543v2) and its leaderboard in 2024.
This meant I did not have to spend as much time I had expected on instrumenting NetHack, but could directly jump to implementing an agent.

Over the course of a week, I 
- first extended the original BALROG Claude/Anthropic integration to use one-shot evaluation (with some history injected as in the original) of each turn and the extended thinking functionality, which was both extremely slow and yielded truly disappointing outcomes, even with thinking budget allowances;
- switched to using the Claude SDK to maintain session continuity, which significantly improved speed, and some initially promising improved outcomes (especially Opus performs automatic preservation of thinking blocks, helping with coherence; this also allowed for much better prompt caching);
- as a final step, switched to using the Claude Agent SDK, providing the previous "text" commands as tools to use, and allowing the agent access to Claude's `TodoWrite` tools, long-running session support with context, etc.
### Tangent: improved observability
I am an impatient person, so I did not want to wait for the run to conclude minutes to hours later, I included journaling of the agent context and logs to a SQLite database that could simultaneously be watched via the browser.

A [video recording of the final play-through playback is available](https://youtu.be/PRQiDtKy5ko). (Though obviously, _something_ had to go wrong, thus this only shows the final todo list and learning of the agent, not the evolution over time; the agent's assistance session is however maintained, and the slow exploration of the dungeon; I did not re-record multiple new runs for time and budget constraints.)

## Conclusions

While achieving *significant* improvements over the BALROG publication and leader board, and fully achieving my personal learning intentions, I'd argue that, given the public claims about "Agentic AI", I would have expected much better. The over-all result seems to independently validate the paper's outcomes and conclusions.

Some tasks that are very hard or require massive effort are trivial and fast for LLMs. From this, we tend to conclude that tasks that are "trivial" and obvious to a human must also be, as we judge the systems by the human difficult scale and by how we would extrapolate from pilots with "traditional" software and algorithms.

This assumption is not correct: human and LLM difficulty scales are *very* different.

What would be somewhat easy for a human, even an untrained one, can be *incredibly* hard to **entirely** out of reach for Gen/AgenticAI LLMs; much less in any way cost-effective or anywhere near acceptable performance. (I am uncertain whether this can be resolved simply through increased scaling, as it seems inherent in the stochastic generation approaches to me.)

The simpler, faster model (Haiku 4.5) was essentially completely unusable and couldn't differentiate between `<` and `>` (which does not bode well for its use with programming language syntax). Even complex and most recent frontier models (Sonnet 4.5, and costly Opus 4.5) were unable to follow key constraints in their instructions, track precise details, following even short-term strategies, much less apply knowledge from their training data. Adherence to explicit and even repeated rules in the prompt is sporadic at best.

The reality is that I'd have likely achieved "better" outcomes (in terms of the achieved NetHack progression score) via a more traditional algorithm and inference engine.

This is not intended to imply that there are not useful applications for LLMs and Generative AI. (I will be writing notes on my experience with using generative AI for the coding aspects of this experiment at a later point.)

Early Gen/Agentic AI wins that appear impressive possibly do not extrapolate to the full problem scope, as we might expect from other problem domains we are familiar with.

*However*, it d·oes suggest certain discrepancies between claims and actual capabilities that cannot be addressed just through the Agent and embedding, but only via future improvements to the algorithms and methods themselves.

## TL;DR

This *was* a lot of fun to work on :-)

### Onwards:

With that, the longer detailed report:


# Agentic Tool Use for Long-Horizon Game Playing

**BRAID** (BALROG Recurrent Agentic Iterative Dungeoneer) — An experimental agent combining Claude's Agent SDK with structured tool use and persistent memory for the NetHack Learning Environment.

---

## Abstract

We present BRAID, an agentic approach to the NetHack Learning Environment (NLE) in the Benchmarking Agentic LLM and VLM Reasoning On Games (BALROG) framework that integrates Anthropic's Claude Agent SDK with Model Context Protocol (MCP) tools and persistent cross-episode memory. Unlike prompt-only approaches that struggle with long-horizon planning in procedurally generated environments, BRAID provides the language model with structured tools for navigation, memory management, and action execution. In single-episode evaluation on the BALROG benchmark, BRAID with Claude Opus 4.5 achieves **6.96% progression** (reaching dungeon level 8), compared to **1.8%** for the best baseline model (Grok-4) on the current leader board. We analyze systemic challenges including glyph interpretation, spatial reasoning, and exploration persistence, and document lessons learned from iterative development including the advantages of native tool use over XML parsing and the importance of explicit constraint enforcement for smaller models.

---

## 1. Introduction

NetHack represents one of the most challenging benchmarks for language model agents. The game combines procedural generation, partial observability, complex item interactions, and permadeath into a task requiring sustained reasoning over thousands of turns. The BALROG benchmark (Benchmarking Agentic LLM and VLM Reasoning On Games) standardizes evaluation across multiple game environments, revealing that current frontier models achieve near-zero progression on NetHack when using standard per-turn prompting approaches.

The fundamental challenge is not necessarily knowledge — language models theoretically possess extensive information about NetHack mechanics from their training data. Rather, the difficulty lies in maintaining coherent strategy across hundreds of game states while parsing ASCII representations and tracking spatial relationships, and applying said theoretical knowledge in practice. A typical successful run would require thousands of decisions, each dependent on correctly interpreting a 24×80 character grid and remembering discoveries made hundreds of turns earlier; and not all scenarios have solutions.

**Contributions.** This work explores whether structured tool use can bridge the gap between model capability and task performance. We introduce:

1. **MCP tool integration** — Navigation, memory, and action tools that provide structured interfaces to game mechanics
2. **Persistent memory** — SQLite-backed storage enabling cross-episode learning
3. **Session continuity** — Maintaining conversation context within episodes while managing token budgets
4. **Safety constraints** — Action queue with automatic abort on combat, HP drops, and traps

Our results suggest that tool-augmented agents significantly outperform prompt-only approaches, though fundamental challenges remain.

---

## 2. Architecture

BRAID extends the BALROG evaluation framework with a custom agent type integrating Anthropic's Claude Agent SDK. The architecture comprises four main components.

### 2.1 Claude Agent SDK Integration

The agent uses Claude's native tool-calling capabilities rather than XML-based action parsing. The `ClaudeToolWrapper` class manages:

- **Session lifecycle** — One SDK session per episode with persistent conversation history
- **MCP server attachment** — Tools registered as `{"braid": mcp_server}` on session initialization
- **PostToolUse hook** — Returns `continue_=False` after game-state-changing tools, forcing return to the game loop for updated observations before further reasoning and decision-making

This design ensures the model always sees current game state before making decisions, preventing cascading errors from stale information.

### 2.2 MCP Tools

Three tool categories expose game functionality:

**Memory tools** (`tools/memory.py`) — `memory_add`, `memory_remove`, `memory_search`, `memory_discover_tags`. Entries have scope (episode or persistent), tags, and priority (1-9). Episode-scoped memories track current-run discoveries (stair locations, room layouts); persistent memories capture cross-game learnings (monster behaviors, mechanics).

**Navigation tools** (`tools/navigation.py`) — `scan` (find monsters/items/exits/unexplored areas), `navigate` (pathfinding queries), `travel_to` (queue movement to coordinates), `travel` (directional movement), `auto_explore` (room perimeter walk with searches).

**Action tool** (`tools/action.py`) — `game_action(*actions)` queues NetHack commands for execution. Supports compound actions ("open north", "kick east") and multi-step sequences.

These were primarily introduced to save on (costly) LLM interactions, but also to see if offloading menial and programmatic tasks would imrpove model outcomes.

#### Claude Built-in tools

In addition to these custom tools, the `TodoWrite` tool was also provided, to assist with the agent's tracking. Claude's models are supposedly trained to leverage this tool, so the model should be capable of using it as needed.

### 2.3 Constraint Enforcement

A critical design constraint: **exactly one game-state-changing tool per turn**. The tools `game_action`, `travel`, `travel_to`, and `auto_explore` all modify game state and thus need to be serialized; only one may be called per LLM invocation (to avoid non-deterministic outcomes, since there is only one player). Read-only tools (`scan`, `memory_*`, `navigate`) have no limit.

This constraint proved essential after observing that Claude Haiku and even Opus frequently attempted to call multiple action tools simultaneously — for example, `travel_to` followed by `game_action` in the same turn. The system detects violations and returns an error message to the model rather than executing conflicting actions.

Notably, they were strongly instructed via the prompt *not* to do this, yet repeatedly violated this rule.

### 2.4 Storage Layer

`BraidStorage` provides unified SQLite storage with WAL mode for multi-worker compatibility. Tables include:

- `memory` — Episode and persistent memories with soft-delete
- `journal` — Timestamped events (requests, responses, tool calls, errors)
- `visited` — Per-level tile visitation tracking keyed by (worker_id, episode, dungeon_num, dlvl)

The visited tile tracking proved critical for exploration planning — without knowing which tiles were already walked, the agent would repeatedly re-explore the same areas. Even with this assistance, it struggled significantly.

`journal` and `visited` are also instrumental in visualizing agent progress for human observation. In hindsight, the `memory` table should also have been versioned by step count for repeated observation of a long-running game session.

### 2.5 Action Queue and Safety Aborts

Multi-step tools like `auto_explore` queue dozens of actions, with the intent of reducing costly API calls as well as demonstrating tool use by the model. The agent executes one action per `act()` call, checking abort conditions between each:

- **HP threshold** — Abort if HP drops below 80% of queue start
- **Hunger worsening** — Abort if hunger state degrades (but pre-existing hunger is tolerated)
- **Combat detected** — Abort on attack messages (excluding pet interactions)
- **Trap triggers** — Abort on trap messages
- **Interactive prompts** — Abort on direction/confirmation prompts requiring decisions

This safety layer prevents the agent from blindly executing queued moves into dangerous situations. Afterwards, observed messages are provided to the assistant en bloc.

---

## 3. Experimental Setup

### 3.1 Environment

Evaluation uses the NetHack Learning Environment (NLE) via the BALROG benchmark's NetHackChallenge-v0 task. The progression metric measures dungeon depth, experience level, and game score, normalized to a 0-100% scale where ascension (winning the game) would score 100%.

### 3.2 Model Configuration

We evaluated three Claude model variants:

| Model             | Thinking Budget |
| ----------------- | --------------- |
| Claude Opus 4.5   | 2048 tokens     |
| Claude Haiku 4.5  | 2048 tokens     |
| Claude Sonnet 4.5 | 2048 tokens     |

**Extended thinking** (thinking_budget) allocates additional tokens for chain-of-thought reasoning before tool calls. Unfortunately, at the time of the experiment, the Agent SDK hides the thinking process ([Agent SDK GitHub issue](https://github.com/anthropics/claude-agent-sdk-typescript/issues/25)), even though it is visible in the regular Claude SDK.

The budget appeared sufficient and to not restrict the models; tentative attempts at using 4096 tokens did not immediately yield any better outcomes.
### 3.3 Prompt Design

The system prompt (`prompt_tools.txt`, 234 lines) follows a hierarchical structure:

1. **Critical constraints** (lines 1-11) — ONE action tool per turn rule
2. **Workflow structure** (lines 12-19) — Systematic turn-by-turn process
3. **Tool documentation** (lines 22-163) — Complete reference with examples
4. **Decision tree** (lines 127-163) — Explicit branching logic for common situations
5. **NetHack essentials** (lines 186-200) — Key mechanics reminders

The decision tree format proved more effective than free-form instructions, providing explicit branches for "in room", "in corridor", "at door", and threat/opportunity assessment.

The prompt was repeatedly iterated over during the experiments, attempting to guide the stochastic generation towards improved outcomes with hopeful incantations; it goes far beyond the shorter prompt of the original BALROG experiments, which likely factors into the stronger results.

---

## 4. Results

### 4.1 Comparison with BALROG Leaderboard

| Agent/Model                  | NLE Progression |
| ---------------------------- | --------------- |
| **BRAID (Opus 4.5)**         | **6.96%**       |
| Grok-4                       | 1.8 ± 0.8%      |
| GPT-5-minimal-think          | 1.3 ± 0.5%      |
| Gemini-2.5-Pro               | 1.7 ± 0.2%      |
| DeepSeek-R1                  | 1.4 ± 0.5%      |
| Claude 3.5 Sonnet (baseline) | 0.6 ± 0.5%      |
| Most other models            | 0.0%            |

BRAID's best and final run achieved significant improvements over the best previously reported score. Note that BRAID results are single-episode (no standard error); baseline results from BALROG leaderboard use multiple episodes.

(Further or even concurrent runs were limited by the available and already significantly exceeded token budget allocated by the author.)

### 4.2 Best Run Analysis

The best run (Opus 4.5, 2048 thinking tokens, 240s timeout) achieved:

- **Progression**: 6.96%
- **Dungeon depth**: Dlvl:8 (main dungeon)
- **Experience level**: XP 5
- **Steps**: 1,763
- **LLM calls**: 298
- **Output tokens**: 100,246
- **Death**: "Killed by a golden naga hatchling, while praying"

The agent explored both the main dungeon (8 levels) and entered the Gnomish Mines branch, though it incorrectly identifyed the mines as a dead-end level ("Mines End") and backtracked to the main dungeon.

**Action distribution** from best run:

| Action | Count | Action | Count |
|--------|-------|--------|-------|
| search | 472 | east | 321 |
| west | 237 | north | 158 |
| south | 154 | southwest | 64 |
| northwest | 59 | northeast | 42 |
| southeast | 39 | far west | 29 |
| far east | 29 | wait | 21 |
| down | 16 | open | 16 |

The high search count (472) reflects the agent's strategy of checking for secret doors — a correct approach, though over-applied. The agent repeatedly, over the course of several observed runs, demonstrated a lack of capability to follow a systematic exploration strategy.

### 4.3 Model Comparison

#### Haiku 4.5

Haiku, Anthropic's smallest and cheapest model, was able to somewhat successfully generate moves, but would almost always try to go down a dungeon level on the `<` symbol (which indicates a staircase to a higher level), indicating even basic glyph differentiation is well beyond it. It'd barely be coherent over even two to three turns, and completely confuse directions.

Thus, Haiku was only used to see if code changes worked, not for any attempt at successfully navigating a dungeon.
#### Sonnet 4.5
The medium Sonnet model performed much better, seemed to exhibit severe limitations with instruction following given the complex constraints as well. Thus, it was not pursued in the interest of achieving a stronger result.
#### Opus 4.5
Opus, the largest and most powerful model, was the only one that appeared capable of a realistic attempt at navigating the dungeon. It occasionally demonstrated that it had previous knowledge from its training data, and sometimes used complex navigation sequences successfully.

However, this comes at significant cost and latency.
#### Context Window Impact
It is worth noting that all compared models had the same context window size (200k tokens), yet demonstrated completely different levels of capabilities.
Thus, mere context window scaling, though often referenced as crucial for long-running or highly complex tasks, is not the sole answer to outcome quality.

---

## 5. Analysis: Systemic Challenges

### 5.1 Glyph Interpretation

NetHack represents the game world as ASCII characters on a 24×80 grid. While language models can identify individual symbols, they struggle with:

- **Relative positioning** — Determining that `@` is adjacent to `+` (closed door) requires parsing row/column coordinates
- **Symbol overloading** — Letters represent both monsters and inventory items depending on context
- **Dynamic state** — The same map position may show `.` (floor), `$` (gold), or a monster letter depending on what's there

The agent frequently attempted invalid actions due to misinterpreting its position relative to targets. e.g., attempting to open a door that was to the north*east* via `open east`, or trying to interact with a feature more than one tile away, failing to consider diagonal navigation. 

#### Why no natural language rendering?
BALROG originally included a "natural" language version of some aspects of the environment; the paper stated that this led to somewhat improved outcomes.
The experiment suggested that this did not hold; in the phase before switching to the Agent SDK which hides the `extended_thinking` blocks, the model was observed to spend significant time trying to reconcile the ASCII map with the simplified language rendering, and no amount of prompt wrangling would get it to simply prioritize the map.
### 5.2 Spatial Reasoning

Even with navigation tools providing pathfinding, the agent exhibited:

- **Coordinate confusion** — Mixing up row/column indices or misremembering positions
- **Direction errors** — Attempting `open north` when the door was `east`, or moving into directions that are clearly blocked on the map
- **Scale misjudgment** — Underestimating distances, leading to interrupted travel commands
- **Difficulty with non-deterministic actions** — the `far ...` movement commands are not entirely predictable in where the player ends up, fast-forwarding through corridors

### 5.3 Exploration Persistence

Maintaining coherent exploration strategy over hundreds of turns proved difficult. Common failure patterns:

- **Incomplete room coverage** — Leaving rooms before fully exploring all exits
- **Corridor amnesia** — Forgetting to return to unexplored corridor branches
- **Level-switching confusion** — Losing track of exploration state after descending and ascending

Anecdotally, the agent searched an isolated floor tile over 100 times across multiple returns to that level, believing stairs must be hidden there. (The stairs were elsewhere entirely.)

The `TodoWrite` tool (inherited from Claude Agent SDK's capabilities) helped somewhat — agents could maintain task lists — but updates were inconsistent, and the agent often failed to consult its own todos when making decisions. Again, no amount of prompting led to it being capable of producing a consistent and complete exploration plan.

### 5.4 Rule and constraint following and learning

Even though the model was clearly instructed to only issue a single game-state modifying command very insistently, all models were observed to do this, and several times during a session (demonstrating they did not "learn" even within a single episode).

Similarly, in NetHack, the player cannot move diagonally through an intact, open door. The model would inevitably try this more than once.

Not even syntax was consistently followed. The original plan for the `game_action()` tool was to accept a list of actions (`game_action("north", "west", "open west")`), but the models would eventually always try `game_action("north, west, open west")`. This syntax error was so persistent that eventually, the tool was adjusted to just accept all common mistakes the model would make.

Neither did they create a persistent memory to aid them in the future for any of these.

Any rule would eventually be observed to be violated, even if explicitly stated and/or repeated.

Even reinjecting the instruction prompt (in addition to the current observations) at certain intervals did not appear to significantly improve this adherence in the limited experiment.

---

## 6. Lessons from Development

### 6.1 XML Parsing to Native Tools

Initial BRAID versions used XML-formatted action specifications parsed from model output, which is the original method how BALROG asked the model to return commands. This approach required ~565 lines of parsing code and was fragile to formatting variations. Switching to Claude Agent's native tool-calling eliminated this complexity entirely and improved reliability. (However, see above: the tools had to be made more resilient with regard to model syntax violations.)

### 6.2 Tool Simplification

Several tools were removed after proving too complex for effective use:

- **Corridor exploration tool** — Combined movement and searching in corridors, but the model couldn't predict or interpret outcomes reliably
- **Visible map tool** — Rendered ASCII visualization of visited/unvisited tiles, but the model struggled to interpret its output

Simpler tools with clearer semantics performed better than sophisticated tools with complex behavior.

### 6.3 Dungeon Branch Tracking

A non-obvious bug emerged: NetHack has multiple dungeon branches (main dungeon, Gnomish Mines, etc.) that share depth numbers. Without branch tracking, the agent would confuse Dlvl:5 of the main dungeon with Dlvl:5 of the Mines. Adding `dungeon_num` to the visited tile schema and announcing branch changes to the agent resolved this.

Even though the prompt reminded the model that multiple branches exist, and this should be well represented in the training data, in no run where the model reached the mines prior to the above addition would it conclude that it entered the mines. Even afterwards, it was unable to fully grasp that those levels do not follow regular room layouts.

(A puzzle level like Sokoban is completely out of reach.)

---

## 7. Emergent Behaviors

### 7.1 Persistent Learning

The agent spontaneously created useful persistent memories, both triggered by observed behaviour and likely informed by some training data:

> "FLOATING EYE: Never melee! Paralyzes on hit. Use ranged/pet/wait."

> "HOMUNCULUS: Puts you to sleep on bite! Very dangerous - multiple attacks while sleeping. Kill fast or avoid."

> "prayer heals to full HP in emergency - use when HP critical"

> "When stairs down not found: search ALL dead-end corridors multiple times (10+ searches each). Secret doors require many searches to find."

These persistent memories were eventually queried at the start of each game, once the agent was explicitly instructed to do so. (It did not ever query the `memory_search` tool spontaneously.)
### 7.2 Episode Memory

The agent maintained structured tracking of dungeon exploration:

> "Reached Dlvl:8 main dungeon. Score 1085. Room has food '%', doors N and W."

> "Dlvl:7 Gnomish Mines may be Mines End or special level - no stairs down found after extensive search. Need to backtrack via stairs up."

> "stairs UP at @65,13 - DO NOT USE (Dlvl:1 exit = loss)"

However, these memories were never queried again for exact coordinates. Instead, the agent relied on its imperfect context.

### 7.3 Strategic Planning

When stuck, the agent would sometimes produce `TodoWrite` plans.

The quality of these plans varied. Sometimes they led to improved coverage, especially in the early game; other times the agent would create a plan but then ignore it.

The instructions in the prompt evolved to give it ever more insistent instructions to make a detailed plan, especially if it got stuck. This did not work.

---

## 8. Limitations and Future Work

### 8.1 Evaluation Scope

Results are based on limited single-episode runs. Statistical significance requires larger-scale evaluation with multiple episodes per configuration, were larger budgets available. The improvement over baselines is promising but should be validated with standard error estimates.

### 8.2 Latency

LLM calls averaged 18-32 seconds with extended thinking enabled. A full episode (1,763 steps, 298 LLM calls) took several hours. This limits iteration speed.

### 8.3 Vision Models

Current implementation uses text-only observation (ASCII grid), after observing that the language rendering made outcomes worse. Vision model integration could potentially improve glyph interpretation by providing the rendered game screen. The BALROG benchmark supports VLM evaluation, showing modest improvements for some models. This was not pursued due to budget limitations.

### 8.4 Future Directions

- **Visualization** — As noted, visual models were not used yet. However, rendering the map as a clear grid (similar to a spreadsheet) would allow much more information to be transmitted (e.g., visibility tracked via background color) in a shape that the model might be better trained on than ASCII renderings.
- **Better per-turn options** — If the natural language observations were congruent with the map observed by the model, and possibly included concrete guidance on _currently_ possible actions (e.g., which directions are blocked), that would possibly improve action choices.
	- _Note 2025-12-08:_ some of this was added, but not fully retested. Initial impressions are promising.
- **Comparison with other Agentic Frameworks** — This focused exclusively on Anthropic's Claude Agent SDK due to time constraints. Comparison with others (such as Google's) would provide additional insights.
- **Better tools** — Perhaps tools are possible that would be better leveraged by the model, offloading more of the deterministic tasks.
- **Programmatic instruction generation** — GenAI models are often trained well in producing (pseudo)code. Perhaps this would be a way of offloading more detailed actions to the agent, while the LLM focuses on "strategy".
- **Multi-episode learning** — Evaluating whether persistent memories actually transfer useful knowledge across games. Given the database backend, these are also immediately sharped across parallel running agents.
- **Knowledge base integration** — Local NetHack Wiki snapshot searchable via tool, providing authoritative mechanics information. At this point, this was very far out of reach and not likely a real limitation.
- **Updating the framework to the latest maintained components and Python version** — This was skipped in the interest of achieving results within the available time. If work were to continue, this would be worthwhile to do.

Note that all of these attempt to reduce the impact of LLM limitations via the agent support.
True fundamentally significant improvements would require the models themselves to get better at instruction following and long-term coherence in complex and dynamic environments.

## References

- [Benchmarking Agentic LLM and VLM Reasoning On Games (Paglierei et al, 2024)](https://arxiv.org/abs/2411.13543v2)
- [BALROG Leaderboard](https://balrogai.com)
- [BALROG GitHub Repository](https://github.com/balrog-ai/BALROG)
- [BALROG Recurrent Agentic Iterative Dungeoneer (Marowsky-Brée, 2025) on GitHub](https://github.com/l-mb/BALROG)
