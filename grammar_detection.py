"""
grammar_detection.py
Grammar Incorrectness Detection module.
Identifies POS patterns in student answers and detects structural violations
without penalizing grammatically imperfect but content-correct answers.
"""

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk, RegexpParser
from collections import defaultdict, Counter
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('chunkers/ne_chunk')
except LookupError:
    nltk.download('maxent_ne_chunker')


class GrammarDetector:
    """
    Detects grammar patterns and violations without harsh penalties.
    Tolerates grammatical imperfections while recognizing content knowledge.
    """
    
    # Common POS tags for reference
    VERB_TAGS = {'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}
    NOUN_TAGS = {'NN', 'NNS', 'NNP', 'NNPS'}
    ADJ_TAGS = {'JJ', 'JJR', 'JJS'}
    ADV_TAGS = {'RB', 'RBR', 'RBS', 'RP'}
    PREP_TAGS = {'IN'}
    CONJ_TAGS = {'CC'}
    
    # Top 50 common adverbs
    COMMON_ADVERBS = {
        'very', 'more', 'most', 'also', 'not', 'just', 'only', 'even', 'already',
        'still', 'too', 'again', 'well', 'now', 'here', 'there', 'where', 'when',
        'why', 'how', 'always', 'never', 'sometimes', 'often', 'usually', 'rarely',
        'quickly', 'slowly', 'suddenly', 'clearly', 'certainly', 'probably', 'possibly',
        'really', 'truly', 'actually', 'basically', 'essentially', 'generally',
        'particularly', 'especially', 'specifically', 'apparently', 'obviously',
        'together', 'apart', 'back', 'forward', 'up', 'down', 'out', 'away', 'around'
    }
    
    # Top 100 common prepositions
    COMMON_PREPOSITIONS = {
        'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'as', 'into',
        'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down',
        'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'about',
        'against', 'between', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'along',
        'across', 'around', 'before', 'behind', 'between', 'beyond', 'during',
        'except', 'inside', 'like', 'near', 'since', 'toward', 'within', 'without',
        'within', 'without', 'throughout', 'upon', 'beneath', 'beside', 'besides',
        'between', 'beyond', 'down', 'during', 'except', 'inside', 'like', 'near',
        'outside', 'since', 'throughout', 'toward', 'towards', 'under', 'underneath',
        'until', 'up', 'upon', 'versus', 'via', 'within', 'without', 'among',
        'amid', 'amongst', 'around', 'before', 'behind', 'below', 'beneath'
    }
    
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.pos_patterns = defaultdict(int)
        self.preposition_usage = defaultdict(int)
        self.adverb_usage = defaultdict(int)
    
    def analyze_pos_structure(self, text: str) -> dict:
        """
        Analyze Part-of-Speech structure of the text.
        """
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        
        # Count POS occurrences
        pos_counts = Counter([pos for _, pos in pos_tags])
        
        # Extract key statistics
        verbs = sum(1 for _, pos in pos_tags if pos in self.VERB_TAGS)
        nouns = sum(1 for _, pos in pos_tags if pos in self.NOUN_TAGS)
        adjectives = sum(1 for _, pos in pos_tags if pos in self.ADJ_TAGS)
        adverbs = sum(1 for _, pos in pos_tags if pos in self.ADV_TAGS)
        prepositions = sum(1 for _, pos in pos_tags if pos in self.PREP_TAGS)
        
        total_tokens = len(tokens)
        
        return {
            'pos_tags': pos_tags,
            'pos_counts': dict(pos_counts),
            'verbs': verbs,
            'nouns': nouns,
            'adjectives': adjectives,
            'adverbs': adverbs,
            'prepositions': prepositions,
            'total_tokens': total_tokens,
            'verb_ratio': verbs / total_tokens if total_tokens > 0 else 0,
            'noun_ratio': nouns / total_tokens if total_tokens > 0 else 0
        }
    
    def detect_grammar_violations(self, text: str) -> dict:
        """
        Detect grammatical violations without being overly strict.
        Returns metrics about grammar quality.
        """
        analysis = self.analyze_pos_structure(text)
        pos_tags = analysis['pos_tags']
        
        violations = {
            'missing_subject_verb': False,
            'run_on_sentence': False,
            'fragment': False,
            'agreement_issues': False,
            'severity': 0,  # 0 = none, 1 = minor, 2 = moderate, 3 = severe
        }
        
        # Check for missing main verb (fragment)
        if analysis['verbs'] == 0 and len(pos_tags) > 3:
            violations['fragment'] = True
            violations['severity'] = max(violations['severity'], 1)
        
        # Check for unbalanced adjectives/adverbs (excessive modifiers)
        modifier_ratio = (analysis['adjectives'] + analysis['adverbs']) / analysis['total_tokens']
        if modifier_ratio > 0.4:  # >40% modifiers is unusual
            violations['run_on_sentence'] = True
            violations['severity'] = max(violations['severity'], 1)
        
        # Check for subject-verb agreement issues
        # Simple heuristic: if we have multiple nouns but no verbs, likely agreement issue
        if analysis['nouns'] > 2 and analysis['verbs'] == 0:
            violations['agreement_issues'] = True
            violations['severity'] = max(violations['severity'], 2)
        
        # No critical violations if we have verbs and reasonable structure
        if analysis['verbs'] > 0 and analysis['noun_ratio'] > 0.1:
            violations['severity'] = 0
        
        return violations
    
    def extract_adverbs(self, text: str) -> dict:
        """
        Extract adverbs from text and check against common patterns.
        """
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        
        found_adverbs = []
        for word, pos in pos_tags:
            if pos in self.ADV_TAGS or word in self.COMMON_ADVERBS:
                found_adverbs.append(word)
                self.adverb_usage[word] += 1
        
        return {
            'adverbs': found_adverbs,
            'adverb_count': len(found_adverbs),
            'common_adverbs': [adv for adv in found_adverbs if adv in self.COMMON_ADVERBS]
        }
    
    def extract_prepositions(self, text: str) -> dict:
        """
        Extract prepositions from text and validate against common patterns.
        """
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        
        found_prepositions = []
        for word, pos in pos_tags:
            if pos in self.PREP_TAGS or word in self.COMMON_PREPOSITIONS:
                found_prepositions.append(word)
                self.preposition_usage[word] += 1
        
        return {
            'prepositions': found_prepositions,
            'preposition_count': len(found_prepositions),
            'valid_prepositions': [prep for prep in found_prepositions if prep in self.COMMON_PREPOSITIONS]
        }
    
    def extract_noun_verb_pairs(self, text: str) -> list:
        """
        Extract noun-verb pairs to understand sentence structure.
        """
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        
        pairs = []
        for i in range(len(pos_tags) - 1):
            word1, pos1 = pos_tags[i]
            word2, pos2 = pos_tags[i + 1]
            
            if pos1 in self.NOUN_TAGS and pos2 in self.VERB_TAGS:
                pairs.append((word1, word2))
        
        return pairs
    
    def assess_grammar_tolerance(self, text: str) -> dict:
        """
        Assess how forgiving we should be of grammar issues.
        Returns metrics that inform whether to apply grammar-related scoring adjustments.
        """
        analysis = self.analyze_pos_structure(text)
        violations = self.detect_grammar_violations(text)
        adverbs = self.extract_adverbs(text)
        prepositions = self.extract_prepositions(text)
        noun_verb_pairs = self.extract_noun_verb_pairs(text)
        
        # Grammar tolerance score: higher = more forgiving
        tolerance_score = 1.0
        
        # Reduce tolerance if severe violations
        if violations['severity'] >= 2:
            tolerance_score *= 0.8
        elif violations['severity'] == 1:
            tolerance_score *= 0.95
        
        # Increase tolerance if structure is reasonable
        if len(noun_verb_pairs) > 0:
            tolerance_score *= 1.05
        
        if analysis['verb_ratio'] > 0.1:  # Has reasonable verb usage
            tolerance_score *= 1.05
        
        # Cap tolerance score
        tolerance_score = min(1.0, max(0.5, tolerance_score))
        
        return {
            'tolerance_score': tolerance_score,
            'violations': violations,
            'adverbs': adverbs,
            'prepositions': prepositions,
            'noun_verb_pairs': noun_verb_pairs,
            'analysis': analysis,
            'recommendation': 'be_lenient' if tolerance_score > 0.85 else 'apply_standard' if tolerance_score > 0.7 else 'apply_strict'
        }
    
    def get_grammar_report(self, text: str) -> str:
        """
        Generate human-readable grammar analysis report.
        """
        assessment = self.assess_grammar_tolerance(text)
        violations = assessment['violations']
        analysis = assessment['analysis']
        
        report = f"""Grammar Analysis Report:
- Tolerance Score: {assessment['tolerance_score']:.2f}
- Recommendation: {assessment['recommendation']}
- Verbs: {analysis['verbs']}, Nouns: {analysis['nouns']}, Adjectives: {analysis['adjectives']}
- Adverbs: {analysis['adverbs']}, Prepositions: {analysis['prepositions']}
- Fragment Detected: {violations['fragment']}
- Agreement Issues: {violations['agreement_issues']}
- Violation Severity: {violations['severity']}/3
- Noun-Verb Pairs: {len(assessment['noun_verb_pairs'])}
"""
        return report
