from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HARNESS = REPO / "evaluation" / "harness"
sys.path.insert(0, str(HARNESS))

import proof_prompts as pp  # noqa: E402
from proof_search import majority_winner  # noqa: E402


class ParseSelectedIdTests(unittest.TestCase):
    def test_well_formed_tag(self):
        self.assertEqual(pp.parse_selected_id("blah <selected_id>P3</selected_id> ok"), "P3")

    def test_last_match_wins(self):
        # the model may reconsider; take its final answer
        self.assertEqual(
            pp.parse_selected_id("<selected_id>P1</selected_id>...<selected_id>P7</selected_id>"),
            "P7",
        )

    def test_open_tag_missing_close(self):
        self.assertEqual(pp.parse_selected_id("... <selected_id>P2 and that's it"), "P2")

    def test_bare_token_last_resort(self):
        self.assertEqual(pp.parse_selected_id("I think the answer is R4."), "R4")

    def test_case_insensitive_and_upper(self):
        self.assertEqual(pp.parse_selected_id("<SELECTED_ID>p5</SELECTED_ID>"), "P5")

    def test_none_when_no_id(self):
        self.assertIsNone(pp.parse_selected_id("no id here at all"))
        self.assertIsNone(pp.parse_selected_id(""))


class SelectionBundleAndMessagesTests(unittest.TestCase):
    def test_bundle_format(self):
        b = pp.selection_bundle([("P1", "proof one"), ("P2", "proof two")])
        self.assertIn('<candidate id="P1">', b)
        self.assertIn('<candidate id="P2">', b)
        self.assertEqual(b.count("<proof>"), 2)
        self.assertEqual(b.count("</candidate>"), 2)
        self.assertIn("proof one", b)
        # order preserved
        self.assertLess(b.index("proof one"), b.index("proof two"))

    def test_selector_messages_render_and_split(self):
        msgs = pp.selector_messages("PROVE X.", pp.selection_bundle([("P1", "the proof body")]))
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        self.assertIn("choosing the final submission", msgs[0]["content"])
        self.assertIn("PROVE X.", msgs[1]["content"])          # {problem} substituted
        self.assertIn("the proof body", msgs[1]["content"])    # {selection_bundle} substituted
        self.assertIn("<selected_id>", msgs[1]["content"])
        # no unreplaced placeholders
        self.assertNotIn("{problem}", msgs[1]["content"])
        self.assertNotIn("{selection_bundle}", msgs[1]["content"])


class MajorityWinnerTests(unittest.TestCase):
    RANK = ["a", "b", "c", "d"]  # rank order (a highest)

    def test_plain_majority(self):
        votes = ["b", "b", "a", "c", "b"]
        self.assertEqual(majority_winner(votes, self.RANK), "b")

    def test_tie_broken_by_rank(self):
        # a and c tie at 2 each; a is higher-ranked -> a wins
        votes = ["c", "a", "c", "a"]
        self.assertEqual(majority_winner(votes, self.RANK), "a")

    def test_nulls_ignored(self):
        self.assertEqual(majority_winner([None, "d", None, "d", "a"], self.RANK), "d")

    def test_all_null_returns_none(self):
        self.assertIsNone(majority_winner([None, None], self.RANK))
        self.assertIsNone(majority_winner([], self.RANK))


class ConfigOptionalSelectorKeysTests(unittest.TestCase):
    def _base_search(self):
        import yaml
        cfg = yaml.safe_load((REPO / "config-nii-r4.yaml").read_text())
        return cfg

    def test_absent_is_valid(self):
        from eval_config import load_config
        import tempfile, yaml
        cfg = self._base_search()
        self.assertNotIn("llm_selector", cfg["search"])  # base config omits it
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(cfg, f)
            path = Path(f.name)
        load_config(path)  # must not raise

    def test_present_and_typed(self):
        from eval_config import load_config
        import tempfile, yaml
        cfg = self._base_search()
        cfg["search"]["llm_selector"] = True
        cfg["search"]["selection_votes"] = 16
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(cfg, f)
            path = Path(f.name)
        loaded = load_config(path)
        self.assertTrue(loaded["search"]["llm_selector"])
        self.assertEqual(loaded["search"]["selection_votes"], 16)

    def test_bad_types_rejected(self):
        from eval_config import load_config
        import tempfile, yaml
        for key, bad in (("llm_selector", "yes"), ("selection_votes", 0), ("selection_votes", -3)):
            cfg = self._base_search()
            cfg["search"][key] = bad
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
                yaml.safe_dump(cfg, f)
                path = Path(f.name)
            with self.assertRaises(ValueError, msg=f"{key}={bad!r} should be rejected"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
