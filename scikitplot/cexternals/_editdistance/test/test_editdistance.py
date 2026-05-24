import unittest
import random

from ..bycython import eval, eval_criterion, eval_dp


class TestEditDistance(unittest.TestCase):
    def test_editdistance(self):
        self.assertEqual(1, eval('abc', 'aec'))

    def test_editdistance_criterion(self):
        self.assertEqual(False, eval_criterion('abcb', 'aeca', 1))
        self.assertEqual(True, eval_criterion('abc', 'aec', 1))

    def test_dp_editdistance(self):
        self.assertEqual(3, eval_dp('bbb', 'a'))
        self.assertEqual(3, eval_dp('a', 'bbb'))

    def test_dp_vs_default(self):
        for _ in range(10):
            seq1 = random.choices([0, 1, 2], k=random.randint(10, 50))
            seq2 = random.choices([0, 1, 2], k=random.randint(10, 50))

            self.assertEqual(eval(seq1, seq2), eval_dp(seq1, seq2))


if __name__ == '__main__':
    unittest.main()
