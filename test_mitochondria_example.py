"""
Test the mitochondria example to see how current implementation handles it.
"""

from content_correctness import ContentCorrectnessChecker
from grammar_detection import GrammarDetector

# Test case from user's example
reference = "the mitochondria produces energy for the cell"
student = "energy the mitochondria cell produces for"

print("=" * 70)
print("MITOCHONDRIA EXAMPLE - EVALUATING ENHANCED IMPLEMENTATION")
print("=" * 70)
print(f"\nReference: {reference}")
print(f"Student:   {student}")
print()

# 1. Test Content Correctness
print("-" * 70)
print("CONTENT CORRECTNESS CHECK")
print("-" * 70)

checker = ContentCorrectnessChecker()

keyword_result = checker.check_keyword_presence(student, reference)
print(f"\nKeyword Match Score: {keyword_result['match_score']:.1%}")
print(f"Content Correct: {keyword_result['content_correct']}")
print(f"Matched Keywords: {keyword_result['matched_keywords']}")
print(f"Missing Keywords: {keyword_result['missing_keywords']}")

concept_result = checker.check_key_concepts(student, reference)
print(f"\nConcept Coverage: {concept_result['concept_coverage']:.1%}")
print(f"Concepts Present: {concept_result['concepts_present']}")
print(f"Matched Concepts: {concept_result['matched_concepts']}")

# NEW: Word Order Detection
print("\n" + "-" * 70)
print("WORD ORDER ANALYSIS (NEW)")
print("-" * 70)

order_result = checker.check_word_order(student, reference)
print(f"\nWord Coverage: {order_result['word_coverage']:.1%}")
print(f"Order Score: {order_result['order_score']:.1%} (lower = more scrambled)")
print(f"Is Scrambled: {order_result['is_scrambled']}")
print(f"Content Preserved: {order_result['content_preserved']}")
print(f"Common Words: {order_result['common_words']}")

# NEW: Content Override Score
print("\n" + "-" * 70)
print("CONTENT OVERRIDE SCORE (NEW)")
print("-" * 70)

override_result = checker.compute_content_override_score(student, reference)
print(f"\nOverride Score: {override_result['override_score']:.1%}")
print(f"Content Score: {override_result['content_score']:.1%}")
print(f"Concept Score: {override_result['concept_score']:.1%}")
print(f"Recommendation: {override_result['recommendation']}")
print(f"Should Boost: {override_result['should_boost']}")
print(f"Boost Amount: {override_result['boost_amount']:.2f}")

# 2. Test Grammar Detection
print("\n" + "-" * 70)
print("GRAMMAR TOLERANCE DETECTION")
print("-" * 70)

detector = GrammarDetector()

# Analyze both reference and student answers
print("\nReference Answer Analysis:")
ref_assessment = detector.assess_grammar_tolerance(reference)
print(f"  Tolerance Score: {ref_assessment['tolerance_score']:.2f}")
print(f"  Verbs: {ref_assessment['analysis']['verbs']}, Nouns: {ref_assessment['analysis']['nouns']}")
print(f"  Violations Severity: {ref_assessment['violations']['severity']}/3")

print("\nStudent Answer Analysis:")
student_assessment = detector.assess_grammar_tolerance(student)
print(f"  Tolerance Score: {student_assessment['tolerance_score']:.2f}")
print(f"  Verbs: {student_assessment['analysis']['verbs']}, Nouns: {student_assessment['analysis']['nouns']}")
print(f"  Violations Severity: {student_assessment['violations']['severity']}/3")
print(f"  Recommendation: {student_assessment['recommendation']}")

# 3. Final Assessment
print("\n" + "=" * 70)
print("FINAL ASSESSMENT")
print("=" * 70)

print(f"\n✓ Content Match Score: {keyword_result['match_score']:.1%}")
print(f"✓ Word Order Score: {order_result['order_score']:.1%}")
print(f"✓ Is Scrambled But Correct: {order_result['is_scrambled']}")
print(f"✓ Override Score: {override_result['override_score']:.1%}")
print(f"✓ Recommendation: {override_result['recommendation']}")

if override_result['should_boost']:
    print("\n→ RESULT: Student demonstrates CLEAR understanding of the concept.")
    print("  All key words present. Content override will boost the score.")
    print(f"  Expected boost: +{override_result['boost_amount']:.1%}")
    if order_result['is_scrambled']:
        print("  Note: Word order is scrambled but content knowledge is evident.")
else:
    print("\n→ Content appears incomplete. Standard scoring applies.")

print("\n" + "=" * 70)
