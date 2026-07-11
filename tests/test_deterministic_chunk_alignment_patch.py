from __future__ import annotations

import unittest

from sglang_patches.patch_deterministic_chunk_alignment import (
    ALIGNMENT_ASSIGNMENT,
    ALIGNMENT_GUARD_MARKER,
    patch_scheduler_text,
)


class DeterministicChunkAlignmentPatchTests(unittest.TestCase):
    def test_patch_is_fail_closed_and_idempotent(self):
        source = "before\n" + ALIGNMENT_ASSIGNMENT + "after\n"
        patched = patch_scheduler_text(source)
        self.assertIn(ALIGNMENT_GUARD_MARKER, patched)
        self.assertIn(
            "chunked_prefill_size < self.truncation_align_size", patched
        )
        self.assertIn("returning OTHER on every scheduling pass", patched)
        self.assertEqual(patch_scheduler_text(patched), patched)

    def test_incomplete_marked_patch_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "guard is incomplete"):
            patch_scheduler_text(ALIGNMENT_GUARD_MARKER + "\n")

    def test_unknown_scheduler_layout_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "source layout changed"):
            patch_scheduler_text("def unrelated():\n    pass\n")


if __name__ == "__main__":
    unittest.main()
