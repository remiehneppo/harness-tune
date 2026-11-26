"""Unit tests for PiCodingAgent NDJSON trace parsing, command isolation, and loop breaking."""

import json
import tempfile
import unittest
from pathlib import Path
from harness_tune.agents.pi_agent import PiCodingAgent, PiNdjsonTraceParser
from harness_tune.core.manifest import HarnessManifest, ModelProfile


class TestPiAgent(unittest.TestCase):
    def test_ndjson_parser_extracts_steps_tools_tokens_and_text(self):
        events = [
            {"type": "session", "id": "abc"},
            {"type": "turn_start"},
            {
                "type": "tool_execution_start",
                "toolCallId": "tc1",
                "toolName": "read",
                "args": {"path": "access.log"},
            },
            {
                "type": "tool_execution_end",
                "toolCallId": "tc1",
                "toolName": "read",
                "result": {"content": [{"type": "text", "text": "line1\nline2"}]},
                "isError": False,
                "durationMs": 12,
            },
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "usage": {"totalTokens": 1079},
                    "stopReason": "toolUse",
                    "content": [{"type": "thinking", "thinking": "Inspecting access.log"}],
                },
            },
            {"type": "turn_end"},
            {"type": "turn_start"},
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "usage": {"totalTokens": 512},
                    "stopReason": "stop",
                    "content": [{"type": "text", "text": "DONE"}],
                },
            },
            {"type": "turn_end"},
            {"type": "agent_end"},
        ]

        parser = PiNdjsonTraceParser(max_steps=25, max_consecutive_tool_errors=2)
        for ev in events:
            stop = parser.process_line(json.dumps(ev))
            self.assertIsNone(stop)

        steps = parser.finalize()
        self.assertEqual(len(steps), 2)
        self.assertEqual(parser.total_tokens, 1591)
        self.assertEqual(parser.final_assistant_text, "DONE")
        self.assertEqual(steps[0].tool_calls[0].tool_name, "read")
        self.assertEqual(steps[0].tool_calls[0].arguments, {"path": "access.log"})
        self.assertEqual(steps[0].tool_calls[0].status, "success")
        self.assertEqual(steps[0].tokens_used, 1079)

    def test_ndjson_parser_triggers_loop_breaker(self):
        parser = PiNdjsonTraceParser(
            max_steps=25,
            max_consecutive_tool_errors=2,
            early_stop_on_repeated_failure=True,
        )
        parser.process_line(json.dumps({"type": "turn_start"}))
        for i in range(2):
            parser.process_line(
                json.dumps(
                    {
                        "type": "tool_execution_start",
                        "toolCallId": f"id_{i}",
                        "toolName": "bash",
                        "args": {"command": "bad_cmd"},
                    }
                )
            )
            stop = parser.process_line(
                json.dumps(
                    {
                        "type": "tool_execution_end",
                        "toolCallId": f"id_{i}",
                        "toolName": "bash",
                        "result": {"content": [{"type": "text", "text": "bash: bad_cmd: command not found"}]},
                        "isError": True,
                        "durationMs": 5,
                    }
                )
            )
            if i == 0:
                self.assertIsNone(stop)
            else:
                self.assertIsNotNone(stop)
                self.assertIn("failed 2 times consecutively", stop)

    def test_build_isolated_command_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            h_dir = Path(tmp)
            skill_file = h_dir / "skills" / "self-evolve" / "SKILL.md"
            skill_file.parent.mkdir(parents=True)
            skill_file.write_text("# Skill\n", encoding="utf-8")

            manifest = HarnessManifest(
                harness_id="pi-arex2",
                version="1.0.0",
                model_profile=ModelProfile(
                    provider="local",
                    model_name="arex-2",
                    thinking_level="medium",
                ),
                surfaces={
                    "system_prompt": "prompts/system.md",
                    "skills": ["skills/self-evolve/SKILL.md"],
                },
                base_dir=h_dir,
            )

            agent = PiCodingAgent(mode="real")
            cmd = agent.build_command(
                pi_bin="/usr/bin/pi",
                instruction="Solve task",
                harness=manifest,
                system_prompt_content="Be concise.",
                runtime_config={"thinking_level": "low"},
                tool_config={"core_tools": ["read", "write", "edit", "bash"]},
                skill_paths=[skill_file],
            )

            for required_flag in [
                "--mode",
                "json",
                "--no-session",
                "--offline",
                "--no-extensions",
                "--no-mcp",
                "--no-context-files",
                "--no-skills",
                "--provider",
                "local",
                "--model",
                "arex-2",
                "--thinking",
                "low",
                "--tools",
                "read,write,edit,bash",
                "--system-prompt",
                "Be concise.",
                "--append-system-prompt",
                str(skill_file),
            ]:
                self.assertIn(required_flag, cmd)
            self.assertEqual(cmd[-2:], ["--", "Solve task"])

    def test_build_user_profile_command_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = HarnessManifest(
                harness_id="pi-arex2-user-default",
                version="1.0.0",
                model_profile=ModelProfile(
                    provider="local",
                    model_name="arex-2",
                    thinking_level="high",
                ),
                surfaces={"system_prompt": "prompts/system.md"},
                base_dir=Path(tmp),
            )
            agent = PiCodingAgent(mode="real", use_user_profile=True)
            cmd = agent.build_command(
                pi_bin="/usr/bin/pi",
                instruction="Solve task",
                harness=manifest,
                system_prompt_content="User SYSTEM.md",
                runtime_config={"use_user_profile": True, "thinking_level": "high"},
                tool_config={"use_default_tools": True},
                skill_paths=[],
                use_user_profile=True,
                skill_dirs=["/opt/harness_pi/user_skills"],
            )
            self.assertNotIn("--no-extensions", cmd)
            self.assertNotIn("--no-skills", cmd)
            self.assertNotIn("--tools", cmd)
            self.assertIn("--skill", cmd)
            self.assertIn("/opt/harness_pi/user_skills", cmd)
            self.assertIn("high", cmd)

    def test_real_mode_errors_when_pi_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = HarnessManifest(
                harness_id="pi-arex2",
                version="1.0.0",
                model_profile=ModelProfile("local", "arex-2"),
                surfaces={},
                base_dir=Path(tmp),
            )
            agent = PiCodingAgent(pi_executable="definitely_missing_pi_binary_xyz", mode="real")
            trace = agent.run_task("t1", "Do work", tmp, manifest)
            self.assertFalse(trace.passed)
            self.assertIn("not found in PATH", trace.error_summary or "")


if __name__ == "__main__":
    unittest.main()
