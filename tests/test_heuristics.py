import unittest
from src.heuristics import (
    clean_text, tfidf_extractive_summary, source_token_overlap,
    extract_medical_entities_regex, extract_concepts, lexicon_concept_f1,
)

class HeuristicTests(unittest.TestCase):
    def test_whitespace_cleaning(self):
        self.assertEqual(clean_text('  One\n  two\tthree  '),'One two three')
        self.assertEqual(clean_text(None),'')
    def test_extraction_uses_source_sentences(self):
        sentences=['The synthetic document describes a bright blue object.',
                   'The bright blue object appears in several photographs.',
                   'A completely unrelated sentence discusses distant mountains.',
                   'Another blue object appears beside the original object.']
        summary=tfidf_extractive_summary(' '.join(sentences),n_sentences=2)
        chosen=[s for s in sentences if s in summary]
        self.assertEqual(len(chosen),2)
        self.assertEqual(summary,' '.join(chosen))
    def test_invalid_sentence_count(self):
        with self.assertRaises(ValueError):tfidf_extractive_summary('text',0)
    def test_token_overlap_copied_text(self):
        self.assertEqual(source_token_overlap('The blue object is visible.','The blue object is visible.'),1)
        self.assertEqual(source_token_overlap('', 'text'),0)
    def test_token_overlap_does_not_check_negation(self):
        # A documented limitation, not clinical validation.
        self.assertEqual(source_token_overlap('Fever reported.','No fever reported.'),1)
    def test_regex_keyword_fallback(self):
        self.assertIn('blood pressure',extract_medical_entities_regex('The note mentions blood pressure.'))
    def test_custom_concept_f1(self):
        self.assertEqual(lexicon_concept_f1('fever pain','pain fever'),1)
        self.assertEqual(lexicon_concept_f1('', 'pain'),0)
    def test_substring_matching_limitation(self):
        self.assertIn('pain',extract_concepts('painting'))

if __name__=='__main__':unittest.main()
