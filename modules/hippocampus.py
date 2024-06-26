import uuid
import time
import joblib
import os
import logging
from typing import List, Dict, Optional, Union
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

class Memory:
    def __init__(self, content: str, tags: List[str] = None, source: str = None, event: str = None):
        self.id = str(uuid.uuid4())
        self.content = content
        self.timestamp = time.time()
        self.tags = tags if tags else []
        self.importance = 1.0
        self.last_accessed = self.timestamp
        self.source = source
        self.event = event

class Hippocampus:
    def __init__(self, capacity: int = 1000, num_clusters: int = 3):
        self.memories: Dict[str, Memory] = {}
        self.tag_index: Dict[str, List[str]] = defaultdict(list)
        self.capacity = capacity
        self.tfidf_vectorizer = TfidfVectorizer()
        self.content_vectors = None
        self.kmeans = KMeans(n_clusters=num_clusters)
        self.memory_clusters = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def store_memory(self, content: str, tags: List[str] = None, source: str = None, event: str = None) -> str:
        try:
            if len(self.memories) >= self.capacity:
                self._forget_least_important_memory()
            
            memory = Memory(content, tags, source, event)
            self.memories[memory.id] = memory
            
            if tags:
                for tag in tags:
                    self.tag_index[tag].append(memory.id)
            
            self._update_tfidf_vectors()
            self._update_clusters()
            return memory.id
        except Exception as e:
            self.logger.error(f"Error storing memory: {str(e)}")
            raise

    def recall_memory(self, query: str, limit: int = 5, tags: List[str] = None, 
                      start_time: float = None, end_time: float = None, source: str = None) -> List[Memory]:
        try:
            if not self.memories:
                return []
            
            query_vector = self.tfidf_vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.content_vectors)[0]
            
            filtered_indices = self._filter_memories(tags, start_time, end_time, source)
            similarities = similarities[filtered_indices]
            
            top_indices = similarities.argsort()[-limit:][::-1]
            relevant_memories = [list(self.memories.values())[filtered_indices[i]] for i in top_indices if similarities[i] > 0]
            
            for memory in relevant_memories:
                self._update_memory_importance(memory.id)
            
            return relevant_memories
        except Exception as e:
            self.logger.error(f"Error recalling memory: {str(e)}")
            raise

    def _filter_memories(self, tags: List[str] = None, start_time: float = None, 
                         end_time: float = None, source: str = None) -> np.ndarray:
        indices = np.arange(len(self.memories))
        memories = list(self.memories.values())
        
        if tags:
            indices = [i for i in indices if any(tag in memories[i].tags for tag in tags)]
        if start_time:
            indices = [i for i in indices if memories[i].timestamp >= start_time]
        if end_time:
            indices = [i for i in indices if memories[i].timestamp <= end_time]
        if source:
            indices = [i for i in indices if memories[i].source == source]
        
        return np.array(indices)

    def _update_tfidf_vectors(self):
        if self.memories:
            contents = [memory.content for memory in self.memories.values()]
            self.tfidf_vectorizer.fit(contents)
            self.content_vectors = self.tfidf_vectorizer.transform(contents)
        else:
            self.content_vectors = None

    def _update_clusters(self):
        if self.content_vectors is not None and self.content_vectors.shape[0] >= self.kmeans.n_clusters:
            self.memory_clusters = self.kmeans.fit_predict(self.content_vectors)
        else:
            self.memory_clusters = np.zeros(len(self.memories))

    def _update_memory_importance(self, memory_id: str):
        memory = self.memories[memory_id]
        current_time = time.time()
        time_factor = np.exp(-0.1 * (current_time - memory.last_accessed) / (24 * 3600))  # Decay over days
        memory.importance *= (1 + time_factor)
        memory.last_accessed = current_time

    def _forget_least_important_memory(self):
        if not self.memories:
            return
        least_important_id = min(self.memories, key=lambda x: self.memories[x].importance)
        forgotten_memory = self.memories.pop(least_important_id)
        for tag in forgotten_memory.tags:
            self.tag_index[tag].remove(least_important_id)
        self._update_tfidf_vectors()
        self._update_clusters()

    def save_state(self, filename: str) -> None:
        try:
            state = {
                'memories': self.memories,
                'tag_index': self.tag_index,
                'capacity': self.capacity,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'kmeans': self.kmeans,
                'memory_clusters': self.memory_clusters
            }
            joblib.dump(state, filename)
            self.logger.info(f"State saved to {filename}")
        except Exception as e:
            self.logger.error(f"Error saving state: {str(e)}")
            raise

    def load_state(self, filename: str) -> None:
        try:
            if not os.path.exists(filename):
                raise FileNotFoundError(f"The file {filename} does not exist.")
            state = joblib.load(filename)
            self.memories = state['memories']
            self.tag_index = state['tag_index']
            self.capacity = state['capacity']
            self.tfidf_vectorizer = state['tfidf_vectorizer']
            self.kmeans = state['kmeans']
            self.memory_clusters = state.get('memory_clusters', None)
            self._update_tfidf_vectors()
            self._update_clusters()
            self.logger.info(f"State loaded from {filename}")
        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}")
            raise

    def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        return self.memories.get(memory_id)

    def update_memory(self, memory_id: str, new_content: str = None, new_tags: List[str] = None, 
                      new_source: str = None, new_event: str = None) -> bool:
        try:
            if memory_id not in self.memories:
                return False
            
            memory = self.memories[memory_id]
            if new_content is not None:
                memory.content = new_content
            if new_tags is not None:
                for old_tag in memory.tags:
                    self.tag_index[old_tag].remove(memory_id)
                memory.tags = new_tags
                for new_tag in new_tags:
                    self.tag_index[new_tag].append(memory_id)
            if new_source is not None:
                memory.source = new_source
            if new_event is not None:
                memory.event = new_event
            
            self._update_tfidf_vectors()
            self._update_clusters()
            return True
        except Exception as e:
            self.logger.error(f"Error updating memory: {str(e)}")
            raise

    def get_memories_by_tag(self, tag: str) -> List[Memory]:
        memory_ids = self.tag_index.get(tag, [])
        return [self.memories[memory_id] for memory_id in memory_ids if memory_id in self.memories]

    def batch_update_memories(self, updates: List[Dict[str, Union[str, List[str]]]]) -> Dict[str, bool]:
        results = {}
        for update in updates:
            memory_id = update.get('id')
            if memory_id:
                results[memory_id] = self.update_memory(
                    memory_id,
                    new_content=update.get('content'),
                    new_tags=update.get('tags'),
                    new_source=update.get('source'),
                    new_event=update.get('event')
                )
        return results

    def summarize_cluster(self, cluster_id: int, max_memories: int = 5) -> str:
        if self.memory_clusters is None:
            self._update_clusters()
        
        if self.memory_clusters is None or cluster_id >= len(np.unique(self.memory_clusters)):
            return "No clusters available."
        
        cluster_memories = [memory for memory, cluster in zip(self.memories.values(), self.memory_clusters) if cluster == cluster_id]
        cluster_memories.sort(key=lambda x: x.importance, reverse=True)
        top_memories = cluster_memories[:max_memories]
        
        summary = f"Cluster {cluster_id} summary:\n"
        for memory in top_memories:
            summary += f"- {memory.content[:100]}...\n"
        
        return summary

    def get_related_memories(self, memory_id: str, limit: int = 5) -> List[Memory]:
        if memory_id not in self.memories or self.memory_clusters is None:
            return []
        
        target_memory = self.memories[memory_id]
        target_cluster = self.memory_clusters[list(self.memories.keys()).index(memory_id)]
        
        related_memories = [memory for memory, cluster in zip(self.memories.values(), self.memory_clusters) 
                            if cluster == target_cluster and memory.id != memory_id]
        related_memories.sort(key=lambda x: x.importance, reverse=True)
        
        return related_memories[:limit]
