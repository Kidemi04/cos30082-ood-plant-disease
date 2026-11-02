from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / 'scripts'))
import evaluate  # noqa: E402
from make_validation_split import choose_validation  # noqa: E402

# Two crops x two diseases; apple_scab, apple_rot and pear_scab are seen, pear_rot is unseen.
LABELS = {
    'crop': {'0': 'apple', '1': 'pear'},
    'disease': {'0': 'scab', '1': 'rot'},
    'composition': {'0': 'apple_scab', '1': 'apple_rot', '2': 'pear_scab', '3': 'pear_rot'},
    'composition_components': {'0': {'crop_id': 0, 'disease_id': 0}, '1': {'crop_id': 0, 'disease_id': 1},
                               '2': {'crop_id': 1, 'disease_id': 0}, '3': {'crop_id': 1, 'disease_id': 1}},
    'seen_composition_ids': [0, 1, 2],
    'unseen_composition_ids': [3],
}
TAXONOMY = evaluate.Taxonomy(LABELS)
PARTS = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}


def records(compositions, prefix='img'):
    return [{'filename': f'{prefix}{i}.jpg', 'composition_id': c, 'crop_id': PARTS[c][0],
             'disease_id': PARTS[c][1]} for i, c in enumerate(compositions)]


def factored(records_, guesses):
    """Crop and disease logits that put 4.0 on each guessed composition's parts."""
    crop, disease = np.zeros((len(guesses), 2)), np.zeros((len(guesses), 2))
    for row, guess in enumerate(guesses):
        crop[row, PARTS[guess][0]] = disease[row, PARTS[guess][1]] = 4.0
    return {'filenames': np.array([r['filename'] for r in records_]), 'crop': crop, 'disease': disease}


class EvaluationTests(unittest.TestCase):
    def test_unseen_pair_is_composed_from_crop_and_disease(self):
        split = records([3, 0])
        metrics, scores, _ = evaluate.evaluate_split(factored(split, [3, 0]), split, TAXONOMY)
        self.assertEqual(scores['composition'].argmax(1).tolist(), [3, 0])
        self.assertEqual(metrics['composition']['top1']['per_image'], 1.0)
        np.testing.assert_allclose(np.exp(scores['composition']).sum(1), 1.0)

    def test_per_class_and_per_image_averages_differ_on_imbalance(self):
        # Four apple_scab images all right, one apple_rot image wrong.
        split = records([0, 0, 0, 0, 1])
        metrics, _, _ = evaluate.evaluate_split(factored(split, [0, 0, 0, 0, 3]), split, TAXONOMY)
        self.assertAlmostEqual(metrics['composition']['top1']['per_image'], 0.8)
        self.assertAlmostEqual(metrics['composition']['top1']['per_class'], 0.5)
        # The wrong guess pear_rot still has the right disease.
        self.assertAlmostEqual(metrics['disease']['top1']['per_image'], 1.0)

    def test_top_k(self):
        scores = np.array([[0.1, 0.5, 0.3, 0.2]])
        self.assertFalse(evaluate.top_k_hits(scores, np.array([2]), 1)[0])
        self.assertTrue(evaluate.top_k_hits(scores, np.array([2]), 2)[0])

    def test_probabilities_and_logits_rank_the_same(self):
        logits = np.array([[2.0, 0.5, -1.0, 0.0], [0.0, 3.0, 1.0, 1.5]])
        probabilities = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        np.testing.assert_allclose(evaluate.log_probabilities(logits), evaluate.log_probabilities(probabilities),
                                   atol=1e-6)

    def test_composition_scores_are_marginalised_to_crop_and_disease(self):
        split = records([1])
        probabilities = np.array([[0.1, 0.5, 0.15, 0.25]])
        predictions = {'filenames': np.array(['img0.jpg']), 'composition': probabilities}
        _, scores, _ = evaluate.evaluate_split(predictions, split, TAXONOMY)
        np.testing.assert_allclose(np.exp(scores['crop']), [[0.6, 0.4]])
        np.testing.assert_allclose(np.exp(scores['disease']), [[0.25, 0.75]])

    def test_harmonic_mean(self):
        self.assertAlmostEqual(evaluate.harmonic_mean(0.8, 0.2), 0.32)
        self.assertEqual(evaluate.harmonic_mean(0.0, 0.0), 0.0)

    def test_report_has_harmonic_mean_and_closed_set_unseen(self):
        seen, unseen = records([0, 1, 2, 2], 'seen'), records([3, 3], 'unseen')
        predictions = {'test_seen': factored(seen, [0, 1, 0, 2]), 'test_unseen': factored(unseen, [3, 1])}
        report, _ = evaluate.evaluate('toy', predictions, {'test_seen': seen, 'test_unseen': unseen}, TAXONOMY)
        self.assertAlmostEqual(report['splits']['test_seen']['composition']['top1']['per_class'], (1 + 1 + 0.5) / 3)
        self.assertAlmostEqual(report['splits']['test_unseen']['composition']['top1']['per_class'], 0.5)
        self.assertAlmostEqual(report['harmonic_mean']['top1']['per_class'],
                               evaluate.harmonic_mean(2.5 / 3, 0.5))
        self.assertEqual(report['splits']['test_unseen']['composition_unseen_only']['top1']['per_image'], 1.0)

    def test_alignment_reorders_and_rejects_mismatches(self):
        split = records([0, 3])
        shuffled = factored(split[::-1], [3, 0])
        metrics, _, _ = evaluate.evaluate_split(shuffled, split, TAXONOMY)
        self.assertEqual(metrics['composition']['top1']['per_image'], 1.0)
        with self.assertRaisesRegex(ValueError, '1 missing'):
            evaluate.align(factored(split[:1], [0]), split)
        duplicated = factored(split, [0, 3])
        duplicated['filenames'][1] = duplicated['filenames'][0]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            evaluate.align(duplicated, split)

    def test_wrong_column_count_is_rejected(self):
        split = records([0])
        predictions = {'filenames': np.array(['img0.jpg']), 'composition': np.zeros((1, 3))}
        with self.assertRaisesRegex(ValueError, '4 columns'):
            evaluate.evaluate_split(predictions, split, TAXONOMY)

    def test_saved_predictions_round_trip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'p.npz'
            evaluate.save_predictions(path, ['a.jpg'], crop=[[1, 2]], disease=[[3, 4]])
            loaded = evaluate.load_predictions(path)
            self.assertEqual(loaded['filenames'].tolist(), ['a.jpg'])
            with self.assertRaisesRegex(ValueError, 'both crop and disease'):
                evaluate.save_predictions(path, ['a.jpg'], crop=[[1, 2]])

    def test_validation_split_is_fixed_stratified_and_skips_rare_compositions(self):
        split = records([0] * 20 + [1] * 2 + [2] * 7)
        chosen = choose_validation(split)
        self.assertEqual(chosen, choose_validation(list(reversed(split))))
        counts = [sum(r['composition_id'] == c for r in chosen) for c in (0, 1, 2)]
        self.assertEqual(counts, [3, 0, 1])


@unittest.skipUnless((PROJECT / 'dataset/prepared/label_map.json').is_file(), 'dataset is not prepared')
class RealDatasetTests(unittest.TestCase):
    def test_oracle_predictions_score_perfectly(self):
        taxonomy = evaluate.Taxonomy(evaluate.load_labels())
        self.assertEqual([taxonomy.size(level) for level in evaluate.LEVELS], [333, 55, 175])
        splits = {split: evaluate.load_split(split) for split in ('test_seen', 'test_unseen')}
        predictions = {}
        for split, rows in splits.items():
            crop, disease = np.zeros((len(rows), 55)), np.zeros((len(rows), 175))
            crop[np.arange(len(rows)), [r['crop_id'] for r in rows]] = 10
            disease[np.arange(len(rows)), [r['disease_id'] for r in rows]] = 10
            predictions[split] = {'filenames': np.array([r['filename'] for r in rows]),
                                  'crop': crop, 'disease': disease}
        report, _ = evaluate.evaluate('oracle', predictions, splits, taxonomy)
        self.assertEqual(report['harmonic_mean']['top1']['per_class'], 1.0)
        self.assertEqual(report['harmonic_mean']['top1']['per_image'], 1.0)


if __name__ == '__main__':
    unittest.main()
