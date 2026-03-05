"""
content_correctness.py
Content Correctness Check module.
Verifies that key words from reference answers appear in student answers,
including synonyms, adjectives, adverbs, and prepositions.
"""

import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import defaultdict
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/universal_tagset')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class ContentCorrectnessChecker:
    """
    Checks whether key words from reference answers appear in student answers,
    including synonyms and related word forms.
    """
    
    def __init__(self):
        self.stopwords = set(stopwords.words('english'))
        self.synonym_cache = {}
        
    def get_synonyms(self, word: str, pos: str = None) -> set:
        """
        Get synonyms for a word using WordNet.
        pos: 'n' (noun), 'v' (verb), 'a' (adjective), 'r' (adverb)
        """
        cache_key = (word.lower(), pos)
        if cache_key in self.synonym_cache:
            return self.synonym_cache[cache_key]
        
        synonyms = {word.lower()}
        
        try:
            pos_map = {'n': wordnet.NOUN, 'v': wordnet.VERB, 'a': wordnet.ADJ, 'r': wordnet.ADV}
            search_pos = [pos_map[pos]] if pos else [wordnet.NOUN, wordnet.VERB, wordnet.ADJ, wordnet.ADV]
            
            for wordnet_pos in search_pos:
                for synset in wordnet.synsets(word, pos=wordnet_pos):
                    for lemma in synset.lemmas():
                        # Add lemma and handle underscores in multi-word synonyms
                        syn = lemma.name().lower().replace('_', ' ')
                        synonyms.add(syn)
        except:
            pass
        
        self.synonym_cache[cache_key] = synonyms
        return synonyms
    
    def extract_keywords(self, text: str, pos_filter: list = None) -> dict:
        """
        Extract keywords from text, optionally filtering by POS.
        Returns dict: {word: (word, pos_tag, is_stopword)}
        pos_filter: list of POS tags to include (e.g., ['NN', 'VB', 'JJ', 'RB'])
        """
        tokens = word_tokenize(text.lower())
        pos_tags = pos_tag(tokens)
        
        keywords = {}
        for word, pos in pos_tags:
            # Skip punctuation and pure numbers
            if not re.match(r'^\w+$', word):
                continue
            
            # Apply POS filter if provided
            if pos_filter and pos not in pos_filter:
                continue
            
            # Skip stopwords unless they're key content words
            is_stop = word in self.stopwords
            
            # Include non-stopwords and key prepositions/conjunctions
            key_function_words = {'is', 'are', 'was', 'were', 'be', 'been', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'of', 'as', 'and', 'or', 'but'}
            if not is_stop or word in key_function_words:
                keywords[word] = (word, pos, is_stop)
        
        return keywords
    
    def check_keyword_presence(self, student_answer: str, reference_answer: str, 
                               threshold: float = 0.6) -> dict:
        """
        Check if key words from reference answer appear in student answer.
        Includes synonym matching.
        
        Returns:
            dict with keys:
                - 'match_score': float 0-1, proportion of reference keywords matched
                - 'matched_keywords': list of (keyword, matched_form) tuples
                - 'missing_keywords': list of unmatched reference keywords
                - 'content_correct': bool, True if match_score >= threshold
        """
        ref_keywords = self.extract_keywords(reference_answer)
        student_text = student_answer.lower()
        student_tokens = set(word_tokenize(student_text.lower()))
        
        matched = []
        missing = []
        
        for keyword, (word, pos, is_stop) in ref_keywords.items():
            # Direct match
            if keyword in student_tokens:
                matched.append((keyword, keyword))
                continue
            
            # Synonym check
            synonyms = self.get_synonyms(keyword, pos=self._pos_to_wordnet(pos))
            found_synonym = False
            
            for syn in synonyms:
                # Check if synonym appears in student answer
                if syn in student_tokens or syn in student_text:
                    matched.append((keyword, syn))
                    found_synonym = True
                    break
            
            if not found_synonym:
                missing.append(keyword)
        
        total_keywords = len(ref_keywords)
        match_score = len(matched) / total_keywords if total_keywords > 0 else 1.0
        
        return {
            'match_score': match_score,
            'matched_keywords': matched,
            'missing_keywords': missing,
            'content_correct': match_score >= threshold,
            'total_keywords': total_keywords,
            'matched_count': len(matched)
        }
    
    def check_key_concepts(self, student_answer: str, reference_answer: str) -> dict:
        """
        Check for presence of key conceptual words (nouns and verbs) from reference.
        More lenient than full keyword check.
        """
        # Extract only nouns (NN*) and verbs (VB*)
        ref_concepts = self.extract_keywords(reference_answer, pos_filter=['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
        student_tokens = set(word_tokenize(student_answer.lower()))
        
        matched_concepts = []
        for concept in ref_concepts.keys():
            if concept in student_tokens:
                matched_concepts.append(concept)
            else:
                # Try synonym matching for concepts
                synonyms = self.get_synonyms(concept)
                for syn in synonyms:
                    if syn in student_tokens or syn in student_answer.lower():
                        matched_concepts.append((concept, syn))
                        break
        
        concept_coverage = len(matched_concepts) / len(ref_concepts) if ref_concepts else 1.0
        
        return {
            'concept_coverage': concept_coverage,
            'matched_concepts': matched_concepts,
            'total_concepts': len(ref_concepts),
            'concepts_present': concept_coverage >= 0.5
        }
    
    def _pos_to_wordnet(self, nltk_pos: str) -> str:
        """Convert NLTK POS tag to WordNet POS tag."""
        if nltk_pos.startswith('NN'):
            return 'n'
        elif nltk_pos.startswith('VB'):
            return 'v'
        elif nltk_pos.startswith('JJ'):
            return 'a'
        elif nltk_pos.startswith('RB'):
            return 'r'
        return None
    
    def check_word_order(self, student_answer: str, reference_answer: str) -> dict:
        """
        Check if words are present but in wrong order.
        Returns metrics about word order correctness.
        """
        from nltk.tokenize import word_tokenize
        
        # Tokenize both answers
        ref_tokens = [w.lower() for w in word_tokenize(reference_answer.lower()) 
                      if w.isalnum() and w.lower() not in self.stopwords]
        student_tokens = [w.lower() for w in word_tokenize(student_answer.lower()) 
                         if w.isalnum() and w.lower() not in self.stopwords]
        
        # Check if same words but different order
        ref_set = set(ref_tokens)
        student_set = set(student_tokens)
        
        # Calculate overlap
        common_words = ref_set & student_set
        word_coverage = len(common_words) / len(ref_set) if ref_set else 0.0
        
        # Check order preservation using longest common subsequence ratio
        def lcs_length(a, b):
            """Compute length of longest common subsequence."""
            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if a[i-1] == b[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            return dp[m][n]
        
        # Only compare common words in their original sequence
        ref_common = [w for w in ref_tokens if w in common_words]
        student_common = [w for w in student_tokens if w in common_words]
        
        if len(ref_common) > 0:
            lcs_len = lcs_length(ref_common, student_common)
            order_score = lcs_len / len(ref_common)
        else:
            order_score = 1.0
        
        # Determine if this is a "scrambled but correct" answer
        is_scrambled = word_coverage >= 0.8 and order_score < 0.6
        content_preserved = word_coverage >= 0.8
        
        return {
            'word_coverage': word_coverage,
            'order_score': order_score,
            'is_scrambled': is_scrambled,
            'content_preserved': content_preserved,
            'common_words': list(common_words),
            'ref_word_count': len(ref_set),
            'student_word_count': len(student_set)
        }
    
    def compute_content_override_score(self, student_answer: str, reference_answer: str) -> dict:
        """
        Compute a content-based score override for answers with correct content
        but poor grammar/word order.
        
        Returns a score adjustment that can boost model predictions when
        content is clearly correct.
        """
        keyword_check = self.check_keyword_presence(student_answer, reference_answer)
        concept_check = self.check_key_concepts(student_answer, reference_answer)
        order_check = self.check_word_order(student_answer, reference_answer)
        
        content_score = keyword_check['match_score']
        concept_score = concept_check['concept_coverage']
        
        # Calculate override score
        # If content is mostly correct (>80%), student demonstrates understanding
        if content_score >= 0.8 or concept_score >= 0.8:
            # Strong content understanding - apply boost
            base_content_score = max(content_score, concept_score)
            
            # If word order is scrambled but content is there, this is a
            # clear case of knowledge + poor grammar
            if order_check['is_scrambled']:
                # Student knows the material but has grammar issues
                # They deserve credit proportional to content correctness
                override_score = base_content_score * 0.85  # Slight penalty for clarity
                recommendation = 'content_boost'
            elif order_check['content_preserved']:
                # Content is there and order is reasonable
                override_score = base_content_score * 0.95
                recommendation = 'full_credit'
            else:
                override_score = base_content_score * 0.75
                recommendation = 'partial_credit'
        else:
            # Content is incomplete
            override_score = content_score * 0.5
            recommendation = 'standard_scoring'
        
        return {
            'override_score': override_score,
            'content_score': content_score,
            'concept_score': concept_score,
            'order_check': order_check,
            'recommendation': recommendation,
            'should_boost': recommendation in ['content_boost', 'full_credit'],
            'boost_amount': max(0, override_score - 0.5) if recommendation != 'standard_scoring' else 0
        }
    
    def get_coverage_report(self, student_answer: str, reference_answer: str) -> str:
        """
        Generate human-readable report of content correctness.
        """
        keyword_check = self.check_keyword_presence(student_answer, reference_answer)
        concept_check = self.check_key_concepts(student_answer, reference_answer)
        
        report = f"""Content Correctness Report:
- Keyword Match Score: {keyword_check['match_score']:.1%} ({keyword_check['matched_count']}/{keyword_check['total_keywords']})
- Concept Coverage: {concept_check['concept_coverage']:.1%} ({len(concept_check['matched_concepts'])}/{concept_check['total_concepts']})
- Content Correct: {keyword_check['content_correct']}
- Concepts Present: {concept_check['concepts_present']}
"""
        
        if keyword_check['missing_keywords']:
            report += f"- Missing Keywords: {', '.join(keyword_check['missing_keywords'][:5])}"
        
        return report
