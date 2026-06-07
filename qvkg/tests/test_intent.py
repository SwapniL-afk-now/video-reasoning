"""Tests for intent classifier."""
import pytest

from qvkg.query.intent import classify_intent


@pytest.mark.parametrize("question,expected_intent", [
    ("Why did she leave the room?",          "CAUSAL"),
    ("When did the explosion happen?",        "TEMPORAL"),
    ("Where is the knife on the table?",      "SPATIAL"),
    ("Who is the person in the red jacket?",  "IDENTITY"),
    ("When did the light turn off?",          "STATE"),
    ("Summarize what happened in the video.", "SUMMARY"),
    ("What if he had not opened the door?",   "COUNTERFACT"),
    ("What kind of scene is this?",           "SEMANTIC"),
])
def test_classify_intent(question, expected_intent):
    intents = classify_intent(question)
    assert expected_intent in intents


def test_fallback_to_semantic():
    # No keywords → falls back to SEMANTIC
    intents = classify_intent("blah blah blah nothing")
    assert intents == ["SEMANTIC"]


def test_multi_intent():
    q = "When and why did the character run away?"
    intents = classify_intent(q)
    assert "TEMPORAL" in intents
    assert "CAUSAL" in intents
