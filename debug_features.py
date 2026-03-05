"""Debug similarity features."""
import backend

backend.initialize_backend()

comprehensive_answer = """The Dynamic Host Configuration Protocol (DHCP) is a network management protocol used in Internet Protocol (IP) networks, whereby a DHCP server dynamically assigns an IP address and other network configuration parameters to each device on the network. DHCP has largely replaced RARP (and BOOTP). Uses of DHCP are: Simplifies installation and configuration of end systems, Allows for manual and automatic IP address assignment, May provide additional configuration information (DNS server, netmask, default router, etc.)"""

question = "What is the Dynamic Host Configuration Protocol (DHCP)? What is it used for?"

reference = backend.REF_LOOKUP.get(question, "")
print(f"Reference found: {len(reference)} chars")
print(f"Reference: {reference[:200]}...")

features = backend.enhanced_similarity(comprehensive_answer, reference)
print(f"\nSimilarity features:")
for k, v in features.items():
    print(f"  {k}: {v:.4f}")

sim_score = backend.reference_similarity(comprehensive_answer, reference)
print(f"\nWeighted similarity: {sim_score:.4f}")

rel = backend.relevance_score(question, comprehensive_answer)
print(f"Relevance score: {rel:.4f}")
