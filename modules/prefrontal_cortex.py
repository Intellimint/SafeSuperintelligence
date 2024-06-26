import numpy as np
from typing import Dict, Any, List
from collections import deque
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os

class PrefrontalCortex:
    def __init__(self, memory_capacity: int = 1000):
        self.knowledge_base: Dict[str, Any] = {}
        self.working_memory: deque = deque(maxlen=memory_capacity)
        self.goals: List[Dict[str, Any]] = []
        self.decision_history: List[Dict[str, Any]] = []
        self.tfidf_vectorizer = TfidfVectorizer()
        self.emotion_state: Dict[str, float] = {"pleasure": 0.0, "arousal": 0.0}
        self.learning_rate: float = 0.1

    def make_decision(self, inputs: Dict[str, Any]) -> str:
        if not isinstance(inputs, dict):
            raise TypeError("Inputs must be a dictionary")
        
        perceived_info = self._process_sensory_input(inputs)
        relevant_knowledge = self._retrieve_relevant_knowledge(perceived_info)
        options = self._generate_options(relevant_knowledge)
        
        if not options:
            return "No viable options found."
        
        decision = self._evaluate_options(options)
        
        self.working_memory.append(decision)
        self.decision_history.append({
            'inputs': inputs,
            'decision': decision,
            'timestamp': self._get_current_timestamp()
        })
        
        self._update_emotion_state(decision)
        return decision

    def _process_sensory_input(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        attention_filter = self._calculate_attention_filter()
        processed = {k: v for k, v in inputs.items() if self._apply_attention_filter(k, attention_filter)}
        return processed if processed else inputs  # Return original inputs if nothing passes the filter

    def _retrieve_relevant_knowledge(self, context: Any) -> List[Any]:
        if isinstance(context, dict):
            context = ' '.join(str(v) for v in context.values())
        if isinstance(context, str) and context.strip():
            if not self.knowledge_base:
                return []
            self._update_tfidf_vectorizer()
            context_vector = self.tfidf_vectorizer.transform([context])
            knowledge_vectors = self.tfidf_vectorizer.transform(list(self.knowledge_base.values()))
            similarities = cosine_similarity(context_vector, knowledge_vectors)[0]
            relevant_indices = np.argsort(similarities)[::-1][:5]  # Top 5 most relevant
            return [list(self.knowledge_base.values())[i] for i in relevant_indices]
        return []

    def _generate_options(self, knowledge: List[Any]) -> List[str]:
        options = []
        for k in knowledge:
            options.extend(self._creative_combination(str(k)))
        return options

    def _evaluate_options(self, options: List[str]) -> str:
        if not options:
            raise ValueError("No options to evaluate")
        scores = [self._score_option(option) for option in options]
        return options[np.argmax(scores)]

    def _integrate_new_knowledge(self, key: str, value: Any) -> None:
        if not key or not value:
            raise ValueError("Key and value must be non-empty")
        existing_value = self.knowledge_base.get(key)
        if existing_value:
            updated_value = self._reconcile_knowledge(existing_value, value)
        else:
            updated_value = value
        self.knowledge_base[key] = updated_value
        self._update_tfidf_vectorizer()

    def _apply_attention_filter(self, key: Any, attention_filter: Dict[str, float]) -> bool:
        return np.random.random() < attention_filter.get(str(key), 0.5)

    def _score_option(self, option: str) -> float:
        goal_alignment = self._calculate_goal_alignment(option)
        emotional_impact = self._calculate_emotional_impact(option)
        return 0.7 * goal_alignment + 0.3 * emotional_impact

    def _calculate_attention_filter(self) -> Dict[str, float]:
        return {goal["goal"]: goal["priority"] for goal in self.goals} if self.goals else {"default": 0.5}

    def _creative_combination(self, knowledge: str) -> List[str]:
        words = knowledge.split()
        return [f"{words[i]} {words[j]}" for i in range(len(words)) for j in range(i+1, len(words))] if len(words) > 1 else [knowledge]

    def _reconcile_knowledge(self, existing: str, new: str) -> str:
        return f"{existing} | {new}"

    def _update_tfidf_vectorizer(self) -> None:
        all_text = list(self.knowledge_base.values()) + [goal["goal"] for goal in self.goals]
        self.tfidf_vectorizer.fit(all_text)

    def _calculate_goal_alignment(self, option: str) -> float:
        if not self.goals or not self.knowledge_base:
            return 0.0
        
        # Ensure the TF-IDF vectorizer is fitted
        if not self.tfidf_vectorizer.vocabulary_:
            self._update_tfidf_vectorizer()
        
        # Transform the option and goals
        option_vector = self.tfidf_vectorizer.transform([option])
        goal_vectors = self.tfidf_vectorizer.transform([goal["goal"] for goal in self.goals])
        
        # Calculate similarities
        similarities = cosine_similarity(option_vector, goal_vectors)[0]
        
        # Calculate the weighted average of similarities
        weights = np.array([goal["priority"] for goal in self.goals])
        weighted_similarity = np.average(similarities, weights=weights)
        
        # Apply a sigmoid function to ensure non-zero output for non-zero input
        alignment = 1 / (1 + np.exp(-10 * (weighted_similarity - 0.5)))
        
        return float(alignment)

    def _calculate_baseline_similarity(self, option: str) -> float:
        # Calculate similarity with a set of common words
        common_words = ["the", "be", "to", "of", "and", "in", "that", "have", "it", "for"]
        common_vector = self.tfidf_vectorizer.transform(common_words)
        option_vector = self.tfidf_vectorizer.transform([option])
        similarities = cosine_similarity(option_vector, common_vector)[0]
        return np.mean(similarities)

    def _calculate_emotional_impact(self, option: str) -> float:
        return (self.emotion_state["pleasure"] + self.emotion_state["arousal"]) / 2

    def _update_emotion_state(self, decision: str) -> None:
        self.emotion_state["pleasure"] += np.random.normal(0, 0.1)
        self.emotion_state["arousal"] += np.random.normal(0, 0.1)
        self.emotion_state = {k: max(min(v, 1), -1) for k, v in self.emotion_state.items()}

    def save_state(self, filename: str) -> None:
        state = {
            'knowledge_base': self.knowledge_base,
            'goals': self.goals,
            'emotion_state': self.emotion_state,
            'learning_rate': self.learning_rate,
            'tfidf_vectorizer': self.tfidf_vectorizer
        }
        joblib.dump(state, filename)

    def load_state(self, filename: str) -> None:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"The file {filename} does not exist.")
        state = joblib.load(filename)
        self.knowledge_base = state.get('knowledge_base', {})
        self.goals = state.get('goals', [])
        self.emotion_state = state.get('emotion_state', {"pleasure": 0.0, "arousal": 0.0})
        self.learning_rate = state.get('learning_rate', 0.1)
        self.tfidf_vectorizer = state.get('tfidf_vectorizer', TfidfVectorizer())

    def _get_current_timestamp(self) -> float:
        return time.time()

    def update_knowledge_base(self, key: str, value: Any) -> None:
        if not key or not value:
            raise ValueError("Key and value must be non-empty")
        self._integrate_new_knowledge(key, value)

    def get_knowledge(self, key: str) -> Any:
        return self.knowledge_base.get(key, "Knowledge not found.")

    def set_goal(self, goal: str, priority: float = 1.0) -> None:
        priority = max(0, min(1, priority))  # Clamp priority between 0 and 1
        self.goals.append({"goal": goal, "priority": priority})
        self.goals.sort(key=lambda x: x["priority"], reverse=True)

    def reflect_on_decision(self, decision: str) -> str:
        # This is a stub for now, to be filled with actual reflection logic
        return f"Reflecting on decision: {decision}"

    def process_feedback(self, response: str, feedback: str) -> None:
        # Example feedback processing
        pass

    def reinforce_behavior(self, response: str) -> None:
        # Example reinforcement logic
        pass

    def adjust_behavior(self, response: str) -> None:
        # Example adjustment logic
        pass
