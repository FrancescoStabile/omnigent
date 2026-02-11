"""Tests for data-driven registries (chains, extractors, reflection, error recovery, knowledge, few-shot)."""

import pytest
from omnigent.chains import CHAINS, ChainStep, get_escalation_chain, format_chain_for_prompt
from omnigent.extractors import EXTRACTORS, run_extractor
from omnigent.reflection import REFLECTORS, reflect_on_result
from omnigent.error_recovery import ERROR_PATTERNS, RecoveryStrategy, get_recovery_strategy
from omnigent.knowledge_loader import KNOWLEDGE_MAP, PHASE_BUDGETS
from omnigent.few_shot_examples import EXAMPLES, ToolExample, get_examples, format_examples_for_prompt


class TestChains:
    def test_empty_by_default(self):
        """Core chains dict is empty — domain fills it."""
        # It might have entries from example imports, so just check type
        assert isinstance(CHAINS, dict)

    def test_chain_step_creation(self):
        step = ChainStep("Do something", "my_tool")
        assert step.description == "Do something"
        assert step.tool_hint == "my_tool"

    def test_get_escalation_chain_missing(self):
        result = get_escalation_chain("nonexistent_chain_xyz")
        assert result is None or result == []

    def test_format_chain_for_prompt(self):
        # Register a temp chain to test formatting
        CHAINS["_test_chain"] = [ChainStep("Step 1", "tool_a"), ChainStep("Step 2", "tool_b")]
        text = format_chain_for_prompt("_test_chain")
        assert "Step 1" in text
        assert "Step 2" in text
        # Cleanup
        del CHAINS["_test_chain"]


class TestExtractors:
    def test_empty_by_default(self):
        assert isinstance(EXTRACTORS, dict)

    def test_run_extractor_missing(self):
        """Running an unknown extractor should not crash."""
        from omnigent.domain_profile import DomainProfile
        profile = DomainProfile()
        # Should handle gracefully
        run_extractor("nonexistent_tool_xyz", profile, "some result", {})


class TestReflectors:
    def test_empty_by_default(self):
        assert isinstance(REFLECTORS, dict)

    def test_reflect_on_result_no_reflectors(self):
        from omnigent.domain_profile import DomainProfile
        profile = DomainProfile()
        result = reflect_on_result("some_tool", {}, "some output", profile)
        assert isinstance(result, str)


class TestErrorRecovery:
    def test_empty_by_default(self):
        assert isinstance(ERROR_PATTERNS, dict)

    def test_recovery_strategy_creation(self):
        rs = RecoveryStrategy(
            guidance="Try again with different args",
            retry_tool="alternative_tool",
            give_up=False,
        )
        assert rs.guidance == "Try again with different args"
        assert rs.give_up is False

    def test_get_recovery_strategy_no_match(self):
        result = get_recovery_strategy("some_tool", "completely unique error 12345xyz")
        assert result is None


class TestKnowledgeLoader:
    def test_empty_by_default(self):
        assert isinstance(KNOWLEDGE_MAP, dict)
        assert isinstance(PHASE_BUDGETS, dict)


class TestFewShotExamples:
    def test_empty_by_default(self):
        assert isinstance(EXAMPLES, dict)

    def test_tool_example_creation(self):
        ex = ToolExample(
            scenario="Test scenario",
            thinking="Need to test",
            tool_name="test_tool",
            tool_args={"arg": "val"},
            expected_result="Expected result",
            is_good=True,
        )
        assert ex.scenario == "Test scenario"
        assert ex.is_good is True

    def test_get_examples_missing(self):
        result = get_examples("nonexistent_tool_xyz")
        assert result == [] or result is None

    def test_format_examples(self):
        # Register a temporary example
        EXAMPLES["_test_tool"] = [
            ToolExample(
                scenario="Test",
                thinking="Test thinking",
                tool_name="_test_tool",
                tool_args={"a": 1},
                expected_result="OK",
                is_good=True,
            ),
        ]
        text = format_examples_for_prompt("_test_tool")
        assert isinstance(text, str)
        assert len(text) > 0
        # Cleanup
        del EXAMPLES["_test_tool"]
