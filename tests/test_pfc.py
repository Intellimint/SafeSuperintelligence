# test_pfc.py

import unittest
from unittest.mock import patch
import time
from parameterized import parameterized
from modules.prefrontal_cortex import PrefrontalCortex
from sklearn.feature_extraction.text import TfidfVectorizer

class TestPrefrontalCortex(unittest.TestCase):
    def setUp(self):
        self.cortex = PrefrontalCortex()

    @parameterized.expand([
        ("Learn AI", 0.8),
        ("Exercise", 0.6),
        ("Read", 0.9),
        ("", 0.5),  # Edge case: empty goal
        ("Test", 1.1),  # Edge case: priority > 1
        ("Another Test", -0.1),  # Edge case: priority < 0
    ])
    def test_set_goal_parameterized(self, goal, priority):
        initial_goal_count = len(self.cortex.goals)
        self.cortex.set_goal(goal, priority)
        self.assertIn(goal, [g['goal'] for g in self.cortex.goals])
        actual_priority = next((g['priority'] for g in self.cortex.goals if g['goal'] == goal), None)
        self.assertIsNotNone(actual_priority)
        self.assertAlmostEqual(actual_priority, max(0, min(1, priority)), places=5)  # Ensure priority is clamped between 0 and 1
        self.assertEqual(len(self.cortex.goals), initial_goal_count + 1)

    def test_update_knowledge_base(self):
        self.cortex.update_knowledge_base("AI", "Artificial Intelligence")
        self.assertIn("AI", self.cortex.knowledge_base)
        self.assertEqual(self.cortex.knowledge_base["AI"], "Artificial Intelligence")

        # Test updating existing knowledge
        self.cortex.update_knowledge_base("AI", "Artificial Intelligence is a field of computer science")
        self.assertEqual(self.cortex.knowledge_base["AI"], "Artificial Intelligence | Artificial Intelligence is a field of computer science")

        # Test with empty key and value
        with self.assertRaises(ValueError):
            self.cortex.update_knowledge_base("", "")

    def test_get_knowledge(self):
        self.cortex.update_knowledge_base("Python", "A programming language")
        self.assertEqual(self.cortex.get_knowledge("Python"), "A programming language")
        self.assertEqual(self.cortex.get_knowledge("Java"), "Knowledge not found.")
        
        # Test with empty key
        self.assertEqual(self.cortex.get_knowledge(""), "Knowledge not found.")

    def test_make_decision_performance(self):
        self.cortex.set_goal("Learn AI", 0.8)
        self.cortex.update_knowledge_base("AI", "Artificial Intelligence")
        
        start_time = time.time()
        decision = self.cortex.make_decision({"context": "study", "topic": "AI"})
        end_time = time.time()
        
        self.assertIsInstance(decision, str)
        self.assertNotEqual(decision, "No viable options found.")
        self.assertLess(end_time - start_time, 1.0)  # Ensure decision is made in less than 1 second

    def test_make_decision_consistency(self):
        self.cortex.set_goal("Learn AI", 0.8)
        self.cortex.update_knowledge_base("AI", "Artificial Intelligence")
        
        decision1 = self.cortex.make_decision({"context": "study", "topic": "AI"})
        decision2 = self.cortex.make_decision({"context": "study", "topic": "AI"})
        
        self.assertEqual(decision1, decision2)  # Ensure consistent output for the same input

    def test_process_sensory_input(self):
        self.cortex.set_goal("Visual processing", 0.8)
        self.cortex.set_goal("Auditory processing", 0.7)
        inputs = {"visual": "red apple", "auditory": "bird chirping"}
        processed = self.cortex._process_sensory_input(inputs)
        self.assertIsInstance(processed, dict)
        self.assertGreater(len(processed), 0)  # At least one input should be processed

    def test_retrieve_relevant_knowledge(self):
        self.cortex.update_knowledge_base("AI", "Artificial Intelligence")
        self.cortex.update_knowledge_base("ML", "Machine Learning")
        relevant = self.cortex._retrieve_relevant_knowledge("AI and ML are related")
        self.assertIsInstance(relevant, list)
        self.assertGreater(len(relevant), 0)

        # Test with empty input
        empty_relevant = self.cortex._retrieve_relevant_knowledge("")
        self.assertEqual(empty_relevant, [])

    def test_generate_options(self):
        options = self.cortex._generate_options(["AI is a field of study"])
        self.assertIsInstance(options, list)
        self.assertGreater(len(options), 0)

        # Test with empty input
        empty_options = self.cortex._generate_options([])
        self.assertEqual(empty_options, [])

    def test_evaluate_options(self):
        self.cortex.set_goal("Learn AI", 0.8)
        self.cortex.update_knowledge_base("AI", "Artificial Intelligence")
        options = ["Study AI", "Take a break", "Read a book"]
        decision = self.cortex._evaluate_options(options)
        self.assertIn(decision, options)

        # Test with empty options
        with self.assertRaises(ValueError):
            self.cortex._evaluate_options([])

    def test_integrate_new_knowledge(self):
        self.cortex._integrate_new_knowledge("AI", "Artificial Intelligence")
        self.assertIn("AI", self.cortex.knowledge_base)

        # Test with empty key and value
        with self.assertRaises(ValueError):
            self.cortex._integrate_new_knowledge("", "")

    @patch('joblib.dump')
    def test_save_state(self, mock_dump):
        self.cortex.save_state("test_state.pkl")
        mock_dump.assert_called_once()

    @patch('joblib.load')
    @patch('os.path.exists', return_value=True)
    def test_load_state(self, mock_exists, mock_load):
        mock_load.return_value = {
            'knowledge_base': {'AI': 'Artificial Intelligence'},
            'goals': [{'goal': 'Learn AI', 'priority': 0.8}],
            'emotion_state': {'pleasure': 0.5, 'arousal': 0.3},
            'learning_rate': 0.1,
            'tfidf_vectorizer': TfidfVectorizer()
        }
        self.cortex.load_state("test_state.pkl")
        self.assertEqual(self.cortex.knowledge_base['AI'], 'Artificial Intelligence')
        self.assertEqual(self.cortex.goals[0]['goal'], 'Learn AI')

    def test_update_emotion_state(self):
        initial_pleasure = self.cortex.emotion_state['pleasure']
        initial_arousal = self.cortex.emotion_state['arousal']
        self.cortex._update_emotion_state("Exciting news!")
        self.assertNotEqual(self.cortex.emotion_state['pleasure'], initial_pleasure)
        self.assertNotEqual(self.cortex.emotion_state['arousal'], initial_arousal)
        
        # Ensure emotion values are clamped between -1 and 1
        self.assertLessEqual(self.cortex.emotion_state['pleasure'], 1)
        self.assertGreaterEqual(self.cortex.emotion_state['pleasure'], -1)
        self.assertLessEqual(self.cortex.emotion_state['arousal'], 1)
        self.assertGreaterEqual(self.cortex.emotion_state['arousal'], -1)

    def test_calculate_goal_alignment(self):
        self.cortex.set_goal("Learn AI", 0.8)
        self.cortex.update_knowledge_base("AI", "Artificial Intelligence")
        alignment = self.cortex._calculate_goal_alignment("Study AI concepts")
        self.assertGreater(alignment, 0)
        self.assertLessEqual(alignment, 1)

        # Test with unrelated input
        unrelated_alignment = self.cortex._calculate_goal_alignment("Go swimming")
        self.assertLess(unrelated_alignment, alignment)

    def test_error_handling(self):
        # Test invalid input types
        with self.assertRaises(TypeError):
            self.cortex.make_decision(123)  # Invalid input type

        with self.assertRaises(TypeError):
            self.cortex.set_goal(123, "Not a number")  # Invalid types for goal and priority

        # Test invalid operations
        with self.assertRaises(ValueError):
            self.cortex._evaluate_options([])  # Empty options list

        # Test accessing non-existent attributes
        with self.assertRaises(AttributeError):
            self.cortex.non_existent_attribute

if __name__ == '__main__':
    unittest.main()
