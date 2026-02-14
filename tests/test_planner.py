"""Tests for the Planner (TaskPlan, TaskPhase, TaskStep)."""

import pytest
from omnigent.planner import (
    TaskPlan, TaskPhase, TaskStep, PhaseStatus,
    PLAN_TEMPLATES, detect_subject_type, generate_plan,
)
from omnigent.domain_profile import DomainProfile


class TestTaskStep:
    def test_defaults(self):
        s = TaskStep(description="Do something")
        assert s.status == PhaseStatus.PENDING
        assert s.tool_hint == ""
        assert s.result_summary == ""


class TestTaskPhase:
    def test_progress(self):
        phase = TaskPhase(
            name="Test",
            objective="Test objective",
            steps=[
                TaskStep(description="A", status=PhaseStatus.COMPLETE),
                TaskStep(description="B", status=PhaseStatus.PENDING),
                TaskStep(description="C", status=PhaseStatus.SKIPPED),
            ],
        )
        assert phase.progress() == "2/3"


class TestTaskPlan:
    def _make_plan(self):
        return TaskPlan(
            objective="Test",
            phases=[
                TaskPhase(
                    name="Phase 1",
                    objective="First",
                    steps=[
                        TaskStep(description="Step 1A", tool_hint="tool_a"),
                        TaskStep(description="Step 1B", tool_hint="tool_b"),
                    ],
                ),
                TaskPhase(
                    name="Phase 2",
                    objective="Second",
                    steps=[
                        TaskStep(description="Step 2A", tool_hint="tool_c"),
                    ],
                ),
            ],
        )

    def test_current_phase_activates_first(self):
        plan = self._make_plan()
        cp = plan.current_phase()
        assert cp is not None
        assert cp.name == "Phase 1"
        assert cp.status == PhaseStatus.ACTIVE

    def test_advance_phase(self):
        plan = self._make_plan()
        plan.current_phase()  # activate Phase 1
        plan.advance_phase()
        assert plan.phases[0].status == PhaseStatus.COMPLETE
        assert plan.phases[1].status == PhaseStatus.ACTIVE

    def test_skip_phase(self):
        plan = self._make_plan()
        plan.skip_phase("Phase 1", "Not applicable")
        assert plan.phases[0].status == PhaseStatus.SKIPPED
        assert all(s.status == PhaseStatus.SKIPPED for s in plan.phases[0].steps)

    def test_mark_step_complete(self):
        plan = self._make_plan()
        plan.current_phase()  # activate
        plan.mark_step_complete("tool_a", "Done")
        assert plan.phases[0].steps[0].status == PhaseStatus.COMPLETE
        assert plan.phases[0].steps[0].result_summary == "Done"
        assert plan.phases[0].steps[1].status == PhaseStatus.PENDING

    def test_mark_step_complete_failure_doesnt_mark(self):
        plan = self._make_plan()
        plan.current_phase()
        plan.mark_step_complete("tool_a", "Error", is_failure=True)
        assert plan.phases[0].steps[0].status == PhaseStatus.PENDING

    def test_is_complete(self):
        plan = self._make_plan()
        assert plan.is_complete() is False
        for phase in plan.phases:
            phase.status = PhaseStatus.COMPLETE
        assert plan.is_complete() is True

    def test_to_prompt_summary(self):
        plan = self._make_plan()
        plan.current_phase()
        summary = plan.to_prompt_summary()
        assert "Phase 1" in summary
        assert "Phase 2" in summary
        assert "INSTRUCTION" in summary

    def test_to_dict_from_dict_roundtrip(self):
        plan = self._make_plan()
        plan.current_phase()
        plan.mark_step_complete("tool_a", "Done")
        d = plan.to_dict()
        restored = TaskPlan.from_dict(d)
        assert restored.objective == plan.objective
        assert len(restored.phases) == 2
        assert restored.phases[0].steps[0].status == PhaseStatus.COMPLETE


class TestFailureTracking:
    """Tests for dynamic replanning: record_failure, reset_failure_count, needs_replan."""

    def _make_plan(self):
        return TaskPlan(
            objective="Test",
            phases=[
                TaskPhase(
                    name="Phase 1",
                    objective="First",
                    steps=[
                        TaskStep(description="Step 1A", tool_hint="tool_a"),
                    ],
                ),
            ],
        )

    def test_consecutive_failures_initialized_zero(self):
        phase = TaskPhase(name="Test", objective="Obj")
        assert phase.consecutive_failures == 0

    def test_record_failure_increments(self):
        plan = self._make_plan()
        plan.current_phase()
        plan.record_failure("tool_a")
        assert plan.phases[0].consecutive_failures == 1
        plan.record_failure("tool_a")
        assert plan.phases[0].consecutive_failures == 2

    def test_reset_failure_count(self):
        plan = self._make_plan()
        plan.current_phase()
        plan.record_failure("tool_a")
        plan.record_failure("tool_a")
        plan.reset_failure_count()
        assert plan.phases[0].consecutive_failures == 0

    def test_needs_replan_below_threshold(self):
        plan = self._make_plan()
        plan.current_phase()
        plan.record_failure("tool_a")
        plan.record_failure("tool_a")
        assert not plan.needs_replan(threshold=3)

    def test_needs_replan_at_threshold(self):
        plan = self._make_plan()
        plan.current_phase()
        for _ in range(3):
            plan.record_failure("tool_a")
        assert plan.needs_replan(threshold=3)

    def test_needs_replan_custom_threshold(self):
        plan = self._make_plan()
        plan.current_phase()
        plan.record_failure("tool_a")
        plan.record_failure("tool_a")
        assert plan.needs_replan(threshold=2)
        assert not plan.needs_replan(threshold=3)

    def test_needs_replan_no_active_phase(self):
        plan = TaskPlan(objective="Empty")
        assert not plan.needs_replan()

    def test_failure_count_serialized_in_to_dict(self):
        plan = self._make_plan()
        plan.current_phase()
        plan.record_failure("tool_a")
        plan.record_failure("tool_a")
        d = plan.to_dict()
        assert d["phases"][0]["consecutive_failures"] == 2

    def test_failure_count_restored_from_dict(self):
        plan = self._make_plan()
        plan.current_phase()
        plan.record_failure("tool_a")
        plan.record_failure("tool_a")
        d = plan.to_dict()
        restored = TaskPlan.from_dict(d)
        assert restored.phases[0].consecutive_failures == 2

    def test_failure_count_defaults_zero_in_from_dict(self):
        """Old serialized plans without consecutive_failures should default to 0."""
        data = {
            "objective": "Test",
            "phases": [
                {
                    "name": "Phase 1",
                    "objective": "First",
                    "steps": [],
                    "status": "active",
                    # No consecutive_failures key
                }
            ],
        }
        plan = TaskPlan.from_dict(data)
        assert plan.phases[0].consecutive_failures == 0


class TestPlanner:
    def test_generate_plan_fallback(self):
        """When no template matches, generates a minimal fallback plan."""
        profile = DomainProfile(subject="test")
        plan = generate_plan("test objective", profile)
        assert plan.objective == "test objective"
        assert len(plan.phases) >= 1

    def test_detect_subject_type_default(self):
        profile = DomainProfile(subject="anything")
        assert detect_subject_type(profile) == "default"
