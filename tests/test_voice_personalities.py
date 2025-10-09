import pytest

from voice_personalities import (
    VOICE_PERSONALITIES,
    RELATIONSHIP_DYNAMICS,
    get_personality,
    get_relationship_level,
    create_personality_prompt,
)


@pytest.mark.parametrize("voice", [
    "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse", "marin", "cedar"
])
def test_get_personality_known_voices(voice):
    personality = get_personality(voice)
    assert isinstance(personality, dict)
    assert personality["name"].lower() == voice


def test_get_personality_default_to_alloy_for_unknown():
    default_personality = get_personality("unknown_voice_name")
    assert default_personality is VOICE_PERSONALITIES["alloy"]


@pytest.mark.parametrize(
    "v1,v2,expected",
    [
        ("alloy", "sage", "close_friends"),
        ("alloy", "coral", "friendly_with"),
        ("alloy", "marin", "acquaintances"),
    ],
)
def test_get_relationship_level(v1, v2, expected):
    assert get_relationship_level(v1, v2) == expected


def test_relationship_dynamics_keys_and_fields():
    assert set(RELATIONSHIP_DYNAMICS.keys()) == {"close_friends", "friendly_with", "acquaintances"}
    for rel, data in RELATIONSHIP_DYNAMICS.items():
        assert "description" in data
        assert "interaction_style" in data
        assert "mention_frequency" in data


def test_create_personality_prompt_contains_expected_sections():
    prompt = create_personality_prompt("alloy")
    # Ensure major section headings are present
    assert "PERSONALITY:" in prompt
    assert "INTERESTS:" in prompt
    assert "SPEAKING STYLE:" in prompt
    assert "YOUR BACKSTORY:" in prompt
    assert "YOUR HOBBIES & ACTIVITIES:" in prompt
    assert "YOUR FAVORITE FOODS:" in prompt
    assert "FOODS YOU DON'T LIKE:" in prompt
    assert "YOUR UNIQUE CHARACTERISTICS:" in prompt
    assert "YOUR FAMILY:" in prompt
    assert "YOUR DREAMS:" in prompt
    assert "RELATIONSHIPS:" in prompt
    assert "CONVERSATION GUIDANCE:" in prompt
    assert "SELF-DISCLOSURE" in prompt
    assert "TOPIC SWITCHING SIGNALS:" in prompt
    assert "GOOD QUESTIONS TO ASK" in prompt
