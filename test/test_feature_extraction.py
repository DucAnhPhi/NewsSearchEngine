import unittest
import json
import os
import sys
sys.path.append("..")
from feature_extraction import FeatureExtraction

class TestFeatureExtraction(unittest.TestCase):

    def test_get_first_paragraph(self):
        actual_text = []
        expected_text = [
            "Die Stadt Karlsruhe will das „Sicherheitsgefühl“ in der Innenstadt verbessern. Die rechtlichen Voraussetzungen für eine polizeiliche Videoüberwachung sind nicht erfüllt. Eine Entwicklung des Energiekonzerns EnBW könnte diese Hürde umgehen.",
            "Der zweite Tag des 32. Chaos Communication Congress ist noch nicht ganz vorüber, aber wir werfen schon mal einen Blick auf die bisherigen Presseberichte und haben ein paar Empfehlungen für den morgigen, dritten Tag zusammengestellt.",
            "Teil I: Es gibt viele Gründe, zum Chaos Communication Congress zu fahren. Ebenso viele sprechen dagegen. Vermutlich stimmt beides. Ein Erlebnisbericht von ebenjenem.",
            "Teil II Was ist das? Was sagt er da? Was machen die da? Der Congress-Neuling schnappt Dinge auf, hört zu und beobachtet. Es ergibt sich ein Text, der nicht all zu leicht zu verstehen ist."
        ]

        with open("test/jsonl/test_articles.jsonl", "r") as f:
            for line in f:
                article = json.loads(line)
                first_paragraph = FeatureExtraction.get_first_paragraph(article)
                actual_text.append(first_paragraph)
        
        for id, actual in enumerate(actual_text):
            self.assertEqual(actual, expected_text[id])