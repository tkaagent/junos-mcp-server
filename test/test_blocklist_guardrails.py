import tempfile
import unittest
from pathlib import Path

from jmcp import check_config_blocklist


class BlocklistGuardrailsTests(unittest.TestCase):
    def test_blocks_literal_prefix_pattern(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            block_file = Path(tmpdir) / "block.cfg"
            block_file.write_text("set system root-authentication\n", encoding="utf-8")

            blocked, message = check_config_blocklist(
                "set system root-authentication encrypted-password foo",
                block_file=str(block_file),
            )

            self.assertTrue(blocked)
            self.assertIn("matches blocked pattern", message)

    def test_blocks_regex_pattern(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            block_file = Path(tmpdir) / "block.cfg"
            block_file.write_text("set system login user (.*) authentication\n", encoding="utf-8")

            blocked, message = check_config_blocklist(
                "set system login user automation authentication encrypted-password bar",
                block_file=str(block_file),
            )

            self.assertTrue(blocked)
            self.assertIn("set system login user (.*) authentication", message)

    def test_blocks_regex_pattern_with_extra_spaces(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            block_file = Path(tmpdir) / "block.cfg"
            block_file.write_text("set system login user (.*) authentication\n", encoding="utf-8")

            blocked, message = check_config_blocklist(
                "set   system   login user guardx authentication encrypted-password xyz",
                block_file=str(block_file),
            )

            self.assertTrue(blocked)
            self.assertIn("set system login user (.*) authentication", message)

    def test_allows_non_blocked_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            block_file = Path(tmpdir) / "block.cfg"
            block_file.write_text("set system root-authentication\n", encoding="utf-8")

            blocked, message = check_config_blocklist(
                "set interfaces ge-0/0/0 description test",
                block_file=str(block_file),
            )

            self.assertFalse(blocked)
            self.assertIsNone(message)


if __name__ == "__main__":
    unittest.main()
