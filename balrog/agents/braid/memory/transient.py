from dataclasses import dataclass, field


@dataclass
class TransientMemory:
    """Per-episode memory that resets between episodes.

    Hybrid structure:
    - Structured: plan, progress, inventory/location notes
    - Free-form: insights list managed by LLM
    """

    current_plan: str = ""
    plan_progress: list[str] = field(default_factory=list)
    inventory_notes: dict[str, str] = field(default_factory=dict)  # item -> note
    location_notes: dict[str, str] = field(default_factory=dict)  # location_key -> note
    insights: list[str] = field(default_factory=list)  # LLM-managed free-form learnings

    def to_prompt_section(self) -> str:
        """Format transient memory for inclusion in prompt."""
        sections = []

        if self.current_plan:
            sections.append(f"CURRENT PLAN:\n{self.current_plan}")
            if self.plan_progress:
                progress = "\n".join(f"- {p}" for p in self.plan_progress[-5:])
                sections.append(f"PROGRESS:\n{progress}")

        if self.inventory_notes:
            notes = "\n".join(f"- {k}: {v}" for k, v in self.inventory_notes.items())
            sections.append(f"INVENTORY NOTES:\n{notes}")

        if self.location_notes:
            notes = "\n".join(f"- {k}: {v}" for k, v in self.location_notes.items())
            sections.append(f"LOCATION NOTES:\n{notes}")

        if self.insights:
            insights = "\n".join(f"- {i}" for i in self.insights[-10:])
            sections.append(f"CURRENT INSIGHTS:\n{insights}")

        return "\n\n".join(sections)

    def reset(self) -> None:
        """Clear all transient memory for new episode."""
        self.current_plan = ""
        self.plan_progress.clear()
        self.inventory_notes.clear()
        self.location_notes.clear()
        self.insights.clear()
