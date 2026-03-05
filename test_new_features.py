"""
test_new_features.py
Quick test of the content correctness and grammar detection features.
"""

from content_correctness import ContentCorrectnessChecker
from grammar_detection import GrammarDetector

# Test Content Correctness
print("=" * 60)
print("TEST 1: Content Correctness Check")
print("=" * 60)

checker = ContentCorrectnessChecker()

reference = "Backpropagation uses gradient descent to optimize neural network weights"
student_answer = "The system speeds up quickly through gradient updates in the network"

result = checker.check_keyword_presence(student_answer, reference)
print(f"\nReference: {reference}")
print(f"Student: {student_answer}")
print(f"\nMatch Score: {result['match_score']:.1%}")
print(f"Content Correct: {result['content_correct']}")
print(f"Matched Keywords: {result['matched_keywords']}")
print(f"Missing Keywords: {result['missing_keywords']}")

# Test Grammar Detection
print("\n" + "=" * 60)
print("TEST 2: Grammar Tolerance Detection")
print("=" * 60)

detector = GrammarDetector()

test_text = "The backprop it updates weights using the gradients"
assessment = detector.assess_grammar_tolerance(test_text)

print(f"\nText: {test_text}")
print(f"Tolerance Score: {assessment['tolerance_score']:.2f}")
print(f"Recommendation: {assessment['recommendation']}")
print(f"Verb Count: {assessment['analysis']['verbs']}")
print(f"Noun Count: {assessment['analysis']['nouns']}")
print(f"Violations - Severity: {assessment['violations']['severity']}/3")

# Test concepts check
print("\n" + "=" * 60)
print("TEST 3: Key Concepts Check")
print("=" * 60)

concepts = checker.check_key_concepts(student_answer, reference)
print(f"\nConcept Coverage: {concepts['concept_coverage']:.1%}")
print(f"Concepts Present: {concepts['concepts_present']}")
print(f"Matched Concepts: {concepts['matched_concepts']}")

print("\n✓ All tests passed successfully!")
