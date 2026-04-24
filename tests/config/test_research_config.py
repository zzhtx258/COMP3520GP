from nanobot.config.schema import ResearchConfig


def test_research_config_dump_uses_public_camel_case_keys() -> None:
    cfg = ResearchConfig.model_validate(
        {
            "maxRounds": 7,
            "maxFindings": 9,
            "maxStaleRounds": 3,
            "model": "openrouter/sonnet",
            "allowWebValidation": False,
        }
    )

    dumped = cfg.model_dump(by_alias=True)

    assert dumped["maxRounds"] == 7
    assert dumped["maxFindings"] == 9
    assert dumped["maxStaleRounds"] == 3
    assert dumped["modelOverride"] == "openrouter/sonnet"
    assert dumped["allowWebValidation"] is False
    assert "model" not in dumped
