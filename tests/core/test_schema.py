from __future__ import annotations

import unittest

from fedicl_mqa.core.schema import label_to_index, normalize_text


class SchemaTests(unittest.TestCase):
    def test_normalization_and_label_parsing(self) -> None:
        self.assertEqual(normalize_text("  Vitamin-C?! "), "vitamin c")
        self.assertEqual(label_to_index("D"), 3)
        self.assertEqual(label_to_index(3), 3)


if __name__ == "__main__":
    unittest.main()
