import unittest
from modules.hippocampus import Hippocampus, Memory
import numpy as np
import time
from unittest.mock import patch, MagicMock
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

class TestHippocampus(unittest.TestCase):
    def setUp(self):
        self.hippocampus = Hippocampus(capacity=3)

    def test_store_memory(self):
        memory_id = self.hippocampus.store_memory("Test memory", tags=["test"])
        self.assertIn(memory_id, self.hippocampus.memories)
        self.assertEqual(self.hippocampus.memories[memory_id].content, "Test memory")

    def test_recall_memory(self):
        self.hippocampus.store_memory("First memory", tags=["first"])
        self.hippocampus.store_memory("Second memory", tags=["second"])
        self.hippocampus.store_memory("Third memory", tags=["third"])

        recalled_memories = self.hippocampus.recall_memory("First")
        self.assertGreater(len(recalled_memories), 0)

    def test_forget_least_important_memory(self):
        memory_id_1 = self.hippocampus.store_memory("First memory")
        memory_id_2 = self.hippocampus.store_memory("Second memory")
        memory_id_3 = self.hippocampus.store_memory("Third memory")
        self.hippocampus.store_memory("Fourth memory")

        self.assertEqual(len(self.hippocampus.memories), 3)
        self.assertNotIn(memory_id_1, self.hippocampus.memories)

    def test_update_memory_importance(self):
        memory_id = self.hippocampus.store_memory("Test memory")
        initial_importance = self.hippocampus.memories[memory_id].importance
        time.sleep(0.1)
        self.hippocampus._update_memory_importance(memory_id)
        updated_importance = self.hippocampus.memories[memory_id].importance
        self.assertGreater(updated_importance, initial_importance)

    @patch('joblib.dump')
    def test_save_state(self, mock_dump):
        self.hippocampus.store_memory("Test memory")
        self.hippocampus.save_state("test_state.pkl")
        mock_dump.assert_called_once()

    @patch('joblib.load')
    @patch('os.path.exists', return_value=True)
    def test_load_state(self, mock_exists, mock_load):
        mock_load.return_value = {
            'memories': {'id1': Memory("Test memory")},
            'tag_index': {'test': ['id1']},
            'capacity': 3,
            'tfidf_vectorizer': TfidfVectorizer(),
            'kmeans': KMeans(n_clusters=10)
        }
        self.hippocampus.load_state("test_state.pkl")
        self.assertIn('id1', self.hippocampus.memories)
        self.assertEqual(self.hippocampus.memories['id1'].content, "Test memory")

    def test_get_memory_by_id(self):
        memory_id = self.hippocampus.store_memory("Test memory")
        memory = self.hippocampus.get_memory_by_id(memory_id)
        self.assertEqual(memory.content, "Test memory")

    def test_update_memory(self):
        memory_id = self.hippocampus.store_memory("Old content", tags=["old"])
        self.hippocampus.update_memory(memory_id, new_content="New content", new_tags=["new"])
        memory = self.hippocampus.get_memory_by_id(memory_id)
        self.assertEqual(memory.content, "New content")
        self.assertIn("new", memory.tags)
        self.assertNotIn("old", memory.tags)

    def test_get_memories_by_tag(self):
        memory_id = self.hippocampus.store_memory("Tagged memory", tags=["test"])
        memories = self.hippocampus.get_memories_by_tag("test")
        self.assertIn(memory_id, [memory.id for memory in memories])

    def test_batch_update_memories(self):
        memory_id1 = self.hippocampus.store_memory("Content 1", tags=["tag1"])
        memory_id2 = self.hippocampus.store_memory("Content 2", tags=["tag2"])
        updates = [
            {'id': memory_id1, 'content': "Updated Content 1", 'tags': ["new_tag1"]},
            {'id': memory_id2, 'content': "Updated Content 2", 'tags': ["new_tag2"]}
        ]
        results = self.hippocampus.batch_update_memories(updates)
        self.assertTrue(results[memory_id1])
        self.assertTrue(results[memory_id2])
        self.assertEqual(self.hippocampus.get_memory_by_id(memory_id1).content, "Updated Content 1")
        self.assertEqual(self.hippocampus.get_memory_by_id(memory_id2).content, "Updated Content 2")

    def test_summarize_cluster(self):
        self.hippocampus.store_memory("Memory 1 in Cluster 1", tags=["cluster1"])
        self.hippocampus.store_memory("Memory 2 in Cluster 1", tags=["cluster1"])
        self.hippocampus._update_clusters()
        summary = self.hippocampus.summarize_cluster(0)
        self.assertIn("Memory 1 in Cluster 1", summary)

    def test_get_related_memories(self):
        memory_id1 = self.hippocampus.store_memory("Memory 1 about AI", tags=["AI"])
        memory_id2 = self.hippocampus.store_memory("Memory 2 about AI", tags=["AI"])
        memory_id3 = self.hippocampus.store_memory("Unrelated memory", tags=["other"])
        self.hippocampus._update_clusters()
        related_memories = self.hippocampus.get_related_memories(memory_id1)
        self.assertGreater(len(related_memories), 0)
        self.assertIn(self.hippocampus.get_memory_by_id(memory_id2), related_memories)
        self.assertNotIn(self.hippocampus.get_memory_by_id(memory_id3), related_memories)

if __name__ == '__main__':
    unittest.main()
