# gen by codeium
import unittest
from rlx.examples.expr_utils import *


class TestDefineRewriteRules(unittest.TestCase):
    def test_r1(self):
        # Testing r1 rule
        self.assertEqual(r1().name, "a+b => b+a")
        self.assertIsNone(r1().a)
        self.assertIsNone(r1().b)
        self.assertEqual(r1().source_pattern(),
                         [node_pattern(node_types["Add"], [None, None], 1)])
        self.assertEqual(r1().target_pattern(),
                         [node_pattern(node_types["Add"], [None, None], 1)])

    def test_r2(self):
        # Testing r2 rule
        self.assertEqual(r2().name, "a*b => b*a")
        self.assertIsNone(r2().a)
        self.assertIsNone(r2().b)
        self.assertEqual(r2().source_pattern(),
                         [node_pattern(node_types["Mul"], [None, None], 1)])
        self.assertEqual(r2().target_pattern(),
                         [node_pattern(node_types["Mul"], [None, None], 1)])

    def test_r3(self):
        # Testing r3 rule
        self.assertEqual(r3().name, "assoc-add")
        self.assertIsNone(r3().a)
        self.assertIsNone(r3().b)
        self.assertIsNone(r3().c)
        self.assertEqual(r3().source_pattern(), [
            node_pattern(node_types["Add"], [None, None], 1),
            node_pattern(node_types["Add"], [None, None], 1)
        ])
        self.assertEqual(r3().target_pattern(), [
            node_pattern(node_types["Add"], [None, None], 1),
            node_pattern(node_types["Add"], [None, None], 1)
        ])

    def test_r4(self):
        # Testing r4 rule
        self.assertEqual(r4().name, "assoc-mul")
        self.assertIsNone(r4().a)
        self.assertIsNone(r4().b)
        self.assertIsNone(r4().c)
        self.assertEqual(r4().source_pattern(), [
            node_pattern(node_types["Mul"], [None, None], 1),
            node_pattern(node_types["Mul"], [None, None], 1)
        ])
        self.assertEqual(r4().target_pattern(), [
            node_pattern(node_types["Mul"], [None, None], 1),
            node_pattern(node_types["Mul"], [None, None], 1)
        ])

    def test_r5(self):
        # Testing r5 rule
        self.assertEqual(r5().name, "sub-canon")
        self.assertIsNone(r5().a)
        self.assertIsNone(r5().b)
        self.assertEqual(r5().source_pattern(),
                         [node_pattern(node_types["Sub"], [None, None], 1)])
        self.assertEqual(r5().target_pattern(), [
            node_pattern(
                node_types["Add"],
                [None, node_pattern(node_types["Mul"], [None, None], 1)], 1)
        ])

    def test_r6(self):
        # Testing r6 rule
        self.assertEqual(r6().name, "zero-add")
        self.assertIsNone(r6().a)
        self.assertEqual(r6().source_pattern(),
                         [node_pattern(node_types["Add"], [None, 0], 1)])
        self.assertEqual(r6().target_pattern(), [None])

    def test_r7(self):
        # Testing r7 rule
        self.assertIsNone(r7().a)
        self.assertEqual(r7().source_pattern(),
                         [node_pattern(node_types["Mul"], [None, 0], 1)])
        self.assertEqual(r7().target_pattern(), [0])

    def test_r8(self):
        # Testing r8 rule
        self.assertIsNone(r8().a)
        self.assertEqual(r8().source_pattern(),
                         [node_pattern(node_types["Mul"], [None, 1], 1)])
        self.assertEqual(r8().target_pattern(), [None])


if __name__ == "__main__":
    unittest.main()
