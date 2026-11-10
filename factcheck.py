import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from typing import List
import numpy as np
import spacy
import gc
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Tokenizes premise and hypothesis, executes a forward pass through the NLI transformer cross-encoder,
        computes softmax class probabilities, and performs explicit tensor cleanup to prevent memory accumulation.
        Returns the probability of entailment (class 0 in MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli).
        """
        with torch.no_grad():
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, max_length=512)
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            # DeBERTa-v3 label order: {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
            p_entailment = float(probs[0].item())

        del inputs, outputs, logits, probs
        gc.collect()

        return p_entailment


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    """
    Part 1: Bag-of-words overlap classifier with morphological stemming,
    stopword filtering, and passage-level word recall thresholding.
    """
    def __init__(self, threshold: float = 0.65, remove_stops: bool = True, stem: bool = True):
        self.threshold = threshold
        self.remove_stops = remove_stops
        self.stem = stem
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))

    def tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r'</?s>', ' ', text).lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', cleaned)
        if self.remove_stops:
            tokens = [t for t in tokens if t not in self.stop_words]
        if self.stem:
            tokens = [self.stemmer.stem(t) for t in tokens]
        return tokens

    def compute_recall(self, fact_tokens: set, passage_text: str) -> float:
        if not fact_tokens:
            return 0.0
        p_tokens = set(self.tokenize(passage_text))
        if not p_tokens:
            return 0.0
        return len(fact_tokens & p_tokens) / len(fact_tokens)

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact_tokens = set(self.tokenize(fact))
        if not fact_tokens:
            return "NS"
        if not passages:
            return "NS"
        max_recall = max((self.compute_recall(fact_tokens, p["text"]) for p in passages), default=0.0)
        return "S" if max_recall >= self.threshold else "NS"


class EntailmentFactChecker(FactChecker):
    """
    Part 2: Textual Entailment Pipeline with Lexical Pruning and DeBERTa Cross-Encoder.
    Incorporates overlap-based preliminary pruning to eliminate irrelevant evidence and
    guarantee fast, memory-safe CPU execution, followed by semantic cross-encoder inference.
    """
    def __init__(self, ent_model,
                 high_recall_threshold: float = 0.82,
                 mod_recall_threshold: float = 0.65,
                 nli_threshold: float = 0.05):
        self.ent_model = ent_model
        self.high_recall_threshold = high_recall_threshold
        self.mod_recall_threshold = mod_recall_threshold
        self.nli_threshold = nli_threshold
        self.stemmer = PorterStemmer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))

    def tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r'</?s>', ' ', text).lower()
        tokens = re.findall(r'\b[a-z0-9]+\b', cleaned)
        return [self.stemmer.stem(t) for t in tokens if t not in self.stop_words]

    def compute_recall(self, fact_tokens: set, passage_text: str) -> float:
        if not fact_tokens:
            return 0.0
        p_tokens = set(self.tokenize(passage_text))
        return len(fact_tokens & p_tokens) / len(fact_tokens) if p_tokens else 0.0

    def predict(self, fact: str, passages: List[dict]) -> str:
        if not fact or not passages:
            return "NS"
        fact_tokens = set(self.tokenize(fact))
        if not fact_tokens:
            return "NS"

        # 1. Lexical Scoring & Pruning across retrieved candidate passages
        passage_scores = []
        for p in passages:
            rec = self.compute_recall(fact_tokens, p["text"])
            passage_scores.append((rec, p))

        passage_scores.sort(key=lambda x: x[0], reverse=True)
        max_recall, best_p = passage_scores[0]

        # Early Pruning: If even the best-matching passage has low keyword overlap,
        # entailment is impossible -> reject immediately without running transformer
        if max_recall < self.mod_recall_threshold:
            return "NS"

        # Decisive Support: Overwhelming keyword overlap indicates clear factual alignment
        if max_recall >= self.high_recall_threshold:
            return "S"

        # 2. Semantic Cross-Encoder Evaluation on top candidates:
        # For moderate overlap claims (0.65 <= recall < 0.85),
        # use DeBERTa to verify whether the passage logically entails the claim
        cand_passages = [p for rec, p in passage_scores[:2] if rec >= self.mod_recall_threshold]
        max_p_entail = 0.0

        for p in cand_passages:
            cleaned = re.sub(r'</?s>', ' ', p["text"]).strip()
            premise = f"{p['title']}. {cleaned}"
            p_ent = self.ent_model.check_entailment(premise, fact)
            if p_ent > max_p_entail:
                max_p_entail = p_ent

        return "S" if max_p_entail >= self.nli_threshold else "NS"


# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations

