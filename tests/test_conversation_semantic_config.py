from __future__ import annotations

from agent.config import load_config
from agent.config_models import ConversationSemanticsConfig


def test_conversation_semantics_config_has_resource_defaults() -> None:
    config = ConversationSemanticsConfig()

    assert config.enabled is True
    assert config.idle_seconds == 480
    assert config.max_turns == 8
    assert config.analysis_version == "conversation-v3"


def test_load_config_reads_conversation_semantics_section(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
provider = "openai"
model = "test-model"

[conversation_semantics]
enabled = false
idle_seconds = 600
max_turns = 12
analysis_version = "conversation-v2"
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.conversation_semantics == ConversationSemanticsConfig(
        enabled=False,
        idle_seconds=600,
        max_turns=12,
        analysis_version="conversation-v2",
    )


def test_load_config_defaults_to_conversation_v3(tmp_path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        'provider = "openai"\nmodel = "test-model"\n',
        encoding="utf-8",
    )

    assert load_config(config_path).conversation_semantics.analysis_version == (
        "conversation-v3"
    )


def test_load_config_reads_langfuse_env_credentials(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://example.langfuse.test")
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
provider = "openai"
model = "test-model"

[observability.langfuse]
enabled = true
public_key = "${LANGFUSE_PUBLIC_KEY}"
secret_key = "${LANGFUSE_SECRET_KEY}"
base_url = "${LANGFUSE_BASE_URL}"
sample_rate = 0.25
capture_content = false
""".strip(),
        encoding="utf-8",
    )

    langfuse = load_config(config_path).observability.langfuse

    assert langfuse.enabled is True
    assert langfuse.public_key == "pk-lf-test"
    assert langfuse.secret_key == "sk-lf-test"
    assert langfuse.base_url == "https://example.langfuse.test"
    assert langfuse.sample_rate == 0.25
    assert langfuse.capture_content is False
