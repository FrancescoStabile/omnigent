"""
Omnigent — Hierarchical Task Planner

Decomposes objectives into structured phases and steps.
Tracks progress and guides the LLM by injecting plan status into context.

The planner does NOT execute — it GUIDES the LLM.

Architecture:
  TaskPlan → TaskPhase → TaskStep
  Each step may have a tool_hint (suggested tool) and status tracking.

Domain customisation:
  1. Define PLAN_TEMPLATES dict with domain-specific phase templates
  2. Override detect_subject_type() to map DomainProfile → template key
  3. Optionally use generate_plan_with_llm() for LLM-refined plans

Example (security domain):
  PLAN_TEMPLATES = {
      "web_standard": [
          {"name": "Discovery", "objective": "...", "steps": [("Port scan", "nmap"), ...]},
          ...
      ],
  }
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from omnigent.domain_profile import DomainProfile

if TYPE_CHECKING:
    from omnigent.router import LLMRouter

logger = logging.getLogger("omnigent.planner")


# ═══════════════════════════════════════════════════════════════════════════
# Status & Data Models
# ═══════════════════════════════════════════════════════════════════════════

class PhaseStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETE = "complete"
    SKIPPED = "skipped"


@dataclass
class TaskStep:
    """A single step within a task phase."""
    description: str
    tool_hint: str = ""
    status: PhaseStatus = PhaseStatus.PENDING
    result_summary: str = ""
    hypothesis_ref: str = ""  # Link to Hypothesis (type:location)


@dataclass
class TaskPhase:
    """A phase of the methodology."""
    name: str
    objective: str
    steps: list[TaskStep] = field(default_factory=list)
    status: PhaseStatus = PhaseStatus.PENDING
    skip_condition: str = ""
    consecutive_failures: int = 0

    def progress(self) -> str:
        total = len(self.steps)
        done = sum(1 for s in self.steps if s.status in (PhaseStatus.COMPLETE, PhaseStatus.SKIPPED))
        return f"{done}/{total}"


@dataclass
class TaskPlan:
    """Full task plan with phases."""
    objective: str
    phases: list[TaskPhase] = field(default_factory=list)
    created_at: str = ""

    def current_phase(self) -> TaskPhase | None:
        """Get the currently active phase, evaluating skip_conditions."""
        for phase in self.phases:
            if phase.status == PhaseStatus.ACTIVE:
                return phase
        # If no phase is active, activate the first pending (skip if condition met)
        for phase in self.phases:
            if phase.status == PhaseStatus.PENDING:
                if phase.skip_condition and self._evaluate_skip_condition(phase):
                    phase.status = PhaseStatus.SKIPPED
                    for step in phase.steps:
                        step.status = PhaseStatus.SKIPPED
                        step.result_summary = f"Skipped: {phase.skip_condition}"
                    continue
                phase.status = PhaseStatus.ACTIVE
                return phase
        return None

    def advance_phase(self):
        """Mark current phase complete and activate next (evaluating skip_conditions)."""
        for i, phase in enumerate(self.phases):
            if phase.status == PhaseStatus.ACTIVE:
                phase.status = PhaseStatus.COMPLETE
                for j in range(i + 1, len(self.phases)):
                    candidate = self.phases[j]
                    if candidate.status != PhaseStatus.PENDING:
                        continue
                    if candidate.skip_condition and self._evaluate_skip_condition(candidate):
                        candidate.status = PhaseStatus.SKIPPED
                        for step in candidate.steps:
                            step.status = PhaseStatus.SKIPPED
                            step.result_summary = f"Skipped: {candidate.skip_condition}"
                        continue
                    candidate.status = PhaseStatus.ACTIVE
                    return
                return

    def skip_phase(self, phase_name: str, reason: str = ""):
        """Skip a phase."""
        for phase in self.phases:
            if phase.name == phase_name:
                phase.status = PhaseStatus.SKIPPED
                for step in phase.steps:
                    step.status = PhaseStatus.SKIPPED
                    step.result_summary = reason or "Skipped"

    def mark_step_complete(self, tool_name: str, result_summary: str = "", *, is_failure: bool = False):
        """Mark a step complete by matching tool hint.

        Only matches steps whose tool_hint is an exact match for the tool name.
        Steps without a tool_hint are left for the LLM to advance manually.
        Failed tool calls do not mark steps complete.
        """
        if is_failure:
            return  # failed tool = step NOT done
        current = self.current_phase()
        if not current:
            return
        for step in current.steps:
            if step.status == PhaseStatus.PENDING and step.tool_hint:
                if step.tool_hint == tool_name:
                    step.status = PhaseStatus.COMPLETE
                    step.result_summary = result_summary[:200]
                    return

    def record_failure(self, tool_name: str) -> None:
        """Record a tool failure in the current phase for replanning detection."""
        current = self.current_phase()
        if current:
            current.consecutive_failures += 1

    def reset_failure_count(self) -> None:
        """Reset failure counter on successful tool execution."""
        current = self.current_phase()
        if current:
            current.consecutive_failures = 0

    def needs_replan(self, threshold: int = 3) -> bool:
        """Check if current phase has too many consecutive failures to warrant replanning."""
        current = self.current_phase()
        return current is not None and current.consecutive_failures >= threshold

    def _evaluate_skip_condition(self, phase: TaskPhase) -> bool:
        """Evaluate a phase's skip_condition against completed phases.

        Skip conditions reference completed phase names or step results.
        Supported patterns:
          - "phase:<name>:complete" — skip if phase <name> is complete
          - "phase:<name>:skipped" — skip if phase <name> was skipped
          - "no_findings" — skip if no steps in previous phases produced results
          - Any other string — treated as keyword search in completed step summaries
        """
        cond = phase.skip_condition.strip().lower()
        if not cond:
            return False

        if cond.startswith("phase:"):
            parts = cond.split(":")
            if len(parts) >= 3:
                ref_name = parts[1]
                ref_status = parts[2]
                for p in self.phases:
                    if p.name.lower() == ref_name:
                        return p.status.value == ref_status
            return False

        if cond == "no_findings":
            for p in self.phases:
                if p.status == PhaseStatus.COMPLETE:
                    for step in p.steps:
                        if step.result_summary and step.status == PhaseStatus.COMPLETE:
                            return False
            return True

        # Keyword search in completed step summaries
        for p in self.phases:
            if p.status == PhaseStatus.COMPLETE:
                for step in p.steps:
                    if cond in step.result_summary.lower():
                        return True
        return False

    def is_complete(self) -> bool:
        return all(p.status in (PhaseStatus.COMPLETE, PhaseStatus.SKIPPED) for p in self.phases)

    def to_prompt_summary(self) -> str:
        """Generate plan summary for LLM context injection."""
        lines = [f"## Task Plan: {self.objective}\n"]

        for i, phase in enumerate(self.phases, 1):
            status_icon = {
                PhaseStatus.PENDING: "[ ]",
                PhaseStatus.ACTIVE: "[>]",
                PhaseStatus.COMPLETE: "[x]",
                PhaseStatus.SKIPPED: "[-]",
            }.get(phase.status, "?")

            lines.append(f"{status_icon} **Phase {i}: {phase.name}** [{phase.progress()}]")
            lines.append(f"  Objective: {phase.objective}")

            if phase.status == PhaseStatus.ACTIVE:
                for j, step in enumerate(phase.steps, 1):
                    step_icon = {
                        PhaseStatus.PENDING: "  [ ]",
                        PhaseStatus.ACTIVE: "  [>]",
                        PhaseStatus.COMPLETE: "  [x]",
                        PhaseStatus.SKIPPED: "  [-]",
                    }.get(step.status, "  ?")
                    lines.append(f"{step_icon} {j}. {step.description}")
                    if step.tool_hint:
                        lines.append(f"       hint: {step.tool_hint}")
                    if step.result_summary:
                        lines.append(f"       result: {step.result_summary}")

        lines.append("")
        lines.append("**INSTRUCTION**: Follow the active phase ([>]). Complete each step, then move to the next.")
        lines.append("If a phase doesn't apply, skip it and explain why.")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> TaskPlan:
        if not data:
            return cls(objective="")
        plan = cls(objective=data.get("objective", ""))
        plan.created_at = data.get("created_at", "")
        for phase_data in data.get("phases", []):
            steps = []
            for s in phase_data.get("steps", []):
                step = TaskStep(
                    description=s.get("description", ""),
                    tool_hint=s.get("tool_hint", ""),
                    status=PhaseStatus(s.get("status", "pending")),
                    result_summary=s.get("result_summary", ""),
                    hypothesis_ref=s.get("hypothesis_ref", ""),
                )
                steps.append(step)
            phase = TaskPhase(
                name=phase_data.get("name", ""),
                objective=phase_data.get("objective", ""),
                steps=steps,
                status=PhaseStatus(phase_data.get("status", "pending")),
                skip_condition=phase_data.get("skip_condition", ""),
                consecutive_failures=phase_data.get("consecutive_failures", 0),
            )
            plan.phases.append(phase)
        return plan


# ═══════════════════════════════════════════════════════════════════════════
# Plan Templates — Override in your domain implementation
# ═══════════════════════════════════════════════════════════════════════════

# Domain implementations populate this with their own templates.
# Each key maps to a list of phase definitions:
#   [{"name": "...", "objective": "...", "steps": [("description", "tool_hint"), ...]}, ...]
#
# Example (security):
#   PLAN_TEMPLATES["web_standard"] = [
#       {"name": "Discovery", "objective": "...", "steps": [("Port scan", "nmap")]},
#       ...
#   ]
PLAN_TEMPLATES: dict[str, list[dict]] = {}


def detect_subject_type(profile: DomainProfile) -> str:
    """Infer the best plan template key from the domain profile.

    Override this in your domain implementation to inspect profile
    metadata/hypotheses and return a template key.

    Returns:
        A key into PLAN_TEMPLATES. Default: 'default'.
    """
    return "default"


def generate_plan(objective: str, profile: DomainProfile) -> TaskPlan:
    """Generate a plan based on objective and profile.

    Uses PLAN_TEMPLATES if available, otherwise returns a minimal plan.
    This is the synchronous fallback when no LLM router is available.
    """
    template_key = detect_subject_type(profile)
    template = PLAN_TEMPLATES.get(template_key)

    if template:
        return _template_to_plan(template_key, objective)

    # Minimal fallback plan
    plan = TaskPlan(objective=objective)
    plan.phases.append(TaskPhase(
        name="Analysis",
        objective="Investigate the subject and gather information",
        steps=[TaskStep(description="Initial analysis of the subject")],
    ))
    plan.phases.append(TaskPhase(
        name="Investigation",
        objective="Deep-dive on identified areas of interest",
        steps=[TaskStep(description="Detailed investigation")],
    ))
    plan.phases.append(TaskPhase(
        name="Results",
        objective="Summarise findings and provide recommendations",
        steps=[TaskStep(description="Generate findings report")],
    ))
    return plan


def _template_to_plan(template_key: str, objective: str) -> TaskPlan:
    """Convert a PLAN_TEMPLATES entry into a live TaskPlan."""
    template = PLAN_TEMPLATES.get(template_key, [])
    plan = TaskPlan(objective=objective)

    for phase_def in template:
        steps = [
            TaskStep(description=desc, tool_hint=hint)
            for desc, hint in phase_def["steps"]
        ]
        plan.phases.append(TaskPhase(
            name=phase_def["name"],
            objective=phase_def["objective"],
            steps=steps,
        ))

    return plan


# ═══════════════════════════════════════════════════════════════════════════
# LLM-Powered Planner
# ═══════════════════════════════════════════════════════════════════════════

# Override this in your domain implementation with domain-specific instructions.
PLANNER_SYSTEM_PROMPT = """\
You are an expert task planner. Given (1) a subject profile and \
(2) a base plan template, return a **refined** JSON plan \
customised for THIS specific subject.

Rules:
- Keep the same phase structure.
- You MAY reorder steps, remove irrelevant ones, or ADD new subject-specific steps.
- Each step is {"description": "...", "tool_hint": "tool_name_or_empty"}.
- Output ONLY valid JSON — no markdown fences, no commentary.
- JSON schema: {"phases": [{"name": str, "objective": str, "steps": [{"description": str, "tool_hint": str}]}]}
"""


PHASE_REFLECTION_PROMPT = """\
You are an expert analyst reviewing the results of a completed phase.

Given the phase details and results below, provide a BRIEF reflection (3-5 sentences):
1. What was accomplished in this phase?
2. Were there any unexpected findings or failures?
3. Should the plan be adjusted for the next phase?
4. What is the most important thing to focus on next?

Output ONLY the reflection text, no JSON or formatting.
"""


async def generate_phase_reflection(
    phase: TaskPhase,
    profile: DomainProfile,
    router: LLMRouter,
) -> str:
    """Generate a macro-reflection summary at the end of a phase.

    Called when advance_phase() completes a phase. The reflection is
    injected into the system prompt for the next phase to guide strategy.

    Returns:
        Reflection text, or empty string on failure.
    """
    from omnigent.router import TaskType

    step_summaries = []
    for step in phase.steps:
        status = step.status.value
        summary = step.result_summary or "no result"
        step_summaries.append(f"  - [{status}] {step.description}: {summary}")

    profile_summary = profile.to_prompt_summary()

    user_prompt = (
        f"## Completed Phase: {phase.name}\n"
        f"**Objective**: {phase.objective}\n\n"
        f"### Step Results:\n" + "\n".join(step_summaries) + "\n\n"
        f"### Current Knowledge:\n{profile_summary}\n\n"
        "Provide your reflection."
    )

    try:
        text = ""
        async with asyncio.timeout(30):
            async for chunk in router.stream(
                messages=[{"role": "user", "content": user_prompt}],
                tools=None,
                system=PHASE_REFLECTION_PROMPT,
                task_type=TaskType.REFLECTION,
            ):
                if chunk.content:
                    text += chunk.content
                if chunk.done:
                    break

        text = text.strip()
        if text:
            logger.info(f"Phase reflection for '{phase.name}': {text[:100]}...")
            return text
    except Exception as e:
        logger.warning(f"Phase reflection failed for '{phase.name}': {e}")

    return ""


async def generate_plan_with_llm(
    objective: str,
    profile: DomainProfile,
    router: LLMRouter,
) -> TaskPlan:
    """Generate a profile-aware plan using LLM refinement.

    1. Detect subject type from profile → select template.
    2. Ask LLM to customise the template for this specific subject.
    3. Parse JSON response → TaskPlan.
    4. Falls back to heuristic template plan on any failure.
    """
    from omnigent.router import TaskType

    subject_type = detect_subject_type(profile)
    base_plan = generate_plan(objective, profile)

    # Serialise template + profile for the LLM
    profile_summary = profile.to_prompt_summary()
    base_plan_json = json.dumps({
        "subject_type": subject_type,
        "phases": [
            {
                "name": p.name,
                "objective": p.objective,
                "steps": [
                    {"description": s.description, "tool_hint": s.tool_hint}
                    for s in p.steps
                ],
            }
            for p in base_plan.phases
        ],
    }, indent=2)

    user_prompt = (
        f"## Objective\n{objective}\n\n"
        f"## Subject Profile\n{profile_summary}\n\n"
        f"## Base Plan (template: {subject_type})\n```json\n{base_plan_json}\n```\n\n"
        "Refine this plan for the specific subject. Return ONLY the JSON."
    )

    try:
        text = ""
        async with asyncio.timeout(90):
            async for chunk in router.stream(
                messages=[{"role": "user", "content": user_prompt}],
                tools=None,
                system=PLANNER_SYSTEM_PROMPT,
                task_type=TaskType.PLANNING,
            ):
                if chunk.content:
                    text += chunk.content
                if chunk.done:
                    break

        # Robust JSON extraction
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            import re as _re
            json_match = _re.search(r'\{[\s\S]*\}', text)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError(f"No JSON object found in LLM response: {text[:200]}")

        plan = TaskPlan(objective=objective)
        for phase_data in data.get("phases", []):
            steps = [
                TaskStep(
                    description=s.get("description", ""),
                    tool_hint=s.get("tool_hint", ""),
                )
                for s in phase_data.get("steps", [])
            ]
            plan.phases.append(TaskPhase(
                name=phase_data.get("name", ""),
                objective=phase_data.get("objective", ""),
                steps=steps,
            ))

        if not plan.phases:
            raise ValueError("LLM returned empty plan")

        logger.info(f"LLM planner: {subject_type} template → {len(plan.phases)} phases, "
                     f"{sum(len(p.steps) for p in plan.phases)} steps")
        return plan

    except Exception as e:
        logger.warning(f"LLM planner failed ({e}), falling back to template '{subject_type}'")
        return base_plan
