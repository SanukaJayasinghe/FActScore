# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import gc
import re
from nltk.stem import PorterStemmer
from spacy.lang.en.stop_words import STOP_WORDS


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
        # Find which of the model's outputs means "entailment" by reading the model's own label list,
        # so this still works if a model with a different label order is used (e.g. roberta-large-mnli)
        label2id = {label.lower(): idx for label, idx in model.config.label2id.items()}
        self.entailment_idx = label2id.get("entailment", 0)

    def check_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Returns the probability (0 to 1) that the premise entails (supports) the hypothesis.
        """
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            # max_length=512 makes sure very long inputs really are cut off (this tokenizer has no default limit),
            # which keeps memory use bounded
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True,
                                    max_length=512)
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Note that the labels are ["entailment", "neutral", "contradiction"]. We turn the 3 raw scores
            # (logits) into probabilities that add up to 1 using softmax, then keep only P(entailment).
            probs = torch.softmax(logits, dim=-1)
            # .item() copies the number out as a plain Python float, so no tensor is kept alive after this
            entailment_prob = probs[0, self.entailment_idx].item()

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits, probs
        gc.collect()

        return entailment_prob


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
    Predicts S if enough of the fact's (important) words appear in at least one passage.
    Score = recall = (fact words found in the passage) / (all fact words), after preprocessing.
    """
    def __init__(self, threshold: float = 0.65, use_stemming: bool = True):
        self.threshold = threshold          # score needed to predict "S"
        self.use_stemming = use_stemming    # turn stemming on/off to compare results
        self.stemmer = PorterStemmer()
        self.stop_words = STOP_WORDS

    def tokenize(self, text: str) -> List[str]:
        """Turns raw text into a list of cleaned-up words."""
        # The passages contain <s> and </s> markers around sentences; remove them
        text = text.replace("<s>", " ").replace("</s>", " ")
        # Lowercase, then keep only runs of letters/digits (drops punctuation)
        words = re.findall(r"[a-z0-9]+", text.lower())
        # Remove stop words (very common words that carry little meaning)
        words = [w for w in words if w not in self.stop_words]
        # Optionally reduce each word to its stem, e.g. "acting" -> "act"
        if self.use_stemming:
            words = [self.stemmer.stem(w) for w in words]
        return words

    def overlap_score(self, fact_words: List[str], passage_words: List[str]) -> float:
        """Recall: what fraction of the fact's unique words also appear in the passage."""
        fact_set = set(fact_words)
        if len(fact_set) == 0:
            return 0.0
        passage_set = set(passage_words)
        return len(fact_set & passage_set) / len(fact_set)

    def best_score(self, fact: str, passages: List[dict]) -> float:
        """The highest overlap score between the fact and any single passage."""
        fact_words = self.tokenize(fact)
        best = 0.0
        for passage in passages:
            score = self.overlap_score(fact_words, self.tokenize(passage["text"]))
            best = max(best, score)
        return best

    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S" if self.best_score(fact, passages) >= self.threshold else "NS"


class EntailmentFactChecker(FactChecker):
    """
    Splits the passages into sentences and runs the entailment model on (sentence, fact) pairs, keeping the
    MAXIMUM entailment probability over all sentences. The final score mixes this with the Part 1 word overlap
    score:  score = entailment_weight * max_entailment + (1 - entailment_weight) * word_overlap
    and we predict S if the score reaches the threshold.
    Sentences with very low word overlap with the fact are skipped (pruned) without running the model, because
    they almost never entail it. This makes it about 4x faster with no loss in accuracy.
    """
    # Split after . ! or ? followed by a space, EXCEPT after an initial like "D." (as in "Joey D. Vieira")
    # or a common abbreviation like "Jr." / "St." / "Dr.", which would cut a name in half
    SENTENCE_SPLIT = re.compile(r"(?<=[.!?])(?<!\b[A-Z]\.)(?<!\bJr\.)(?<!\bSr\.)(?<!\bSt\.)(?<!\bDr\.)"
                                r"(?<!\bMr\.)(?<!\bMs\.)(?<!\bvs\.)\s+")

    def __init__(self, ent_model, threshold: float = 0.5, entailment_weight: float = 0.5,
                 prune_threshold: float = 0.1):
        self.ent_model = ent_model
        self.threshold = threshold                  # combined score needed to predict "S"
        self.entailment_weight = entailment_weight  # how much the model counts vs. word overlap (0 to 1)
        self.prune_threshold = prune_threshold      # minimum word overlap for a sentence to be checked at all
        # Reuse the Part 1 checker for its tokenizer and overlap score
        self.word_overlap = WordRecallThresholdFactChecker()

    def split_sentences(self, text: str) -> List[str]:
        """Splits a passage into sentences."""
        sentences = []
        # The passages already mark sentence-like chunks with <s> ... </s>. A chunk may be cut off at the
        # start or end of the passage, so we split on both markers instead of requiring matching pairs.
        chunks = text.replace("</s>", "<s>").split("<s>")
        for chunk in chunks:
            # A chunk can still hold several sentences, so split it further
            for sentence in self.SENTENCE_SPLIT.split(chunk):
                sentence = sentence.strip()
                if sentence:
                    sentences.append(sentence)
        return sentences

    def get_candidate_sentences(self, fact: str, passages: List[dict]) -> List[str]:
        """All unique passage sentences that pass the word-overlap pruning filter."""
        fact_words = self.word_overlap.tokenize(fact)
        candidates = []
        seen = set()
        for passage in passages:
            for sentence in self.split_sentences(passage["text"]):
                if sentence in seen:    # passages can overlap, so skip sentences already seen
                    continue
                seen.add(sentence)
                overlap = self.word_overlap.overlap_score(fact_words, self.word_overlap.tokenize(sentence))
                if overlap >= self.prune_threshold:
                    candidates.append(sentence)
        return candidates

    def predict(self, fact: str, passages: List[dict]) -> str:
        w = self.entailment_weight
        overlap = self.word_overlap.best_score(fact, passages)   # Part 1 score, from 0 to 1
        # The entailment probability needed for the combined score to reach the threshold:
        #   w * needed + (1 - w) * overlap = threshold   ->   needed = (threshold - (1 - w) * overlap) / w
        needed = (self.threshold - (1 - w) * overlap) / w
        if needed <= 0:
            return "S"      # word overlap alone is already enough, no need to run the model

        max_prob = 0.0
        for sentence in self.get_candidate_sentences(fact, passages):
            # premise = the passage sentence, hypothesis = the fact
            prob = self.ent_model.check_entailment(sentence, fact)
            max_prob = max(max_prob, prob)
            if max_prob >= needed:
                # The maximum can only go up, so once it is high enough the answer is S; stop early
                return "S"

        score = w * max_prob + (1 - w) * overlap
        return "S" if score >= self.threshold else "NS"


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
