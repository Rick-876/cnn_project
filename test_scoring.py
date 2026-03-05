"""Test the calibrated scoring."""
import requests
import json

# Test comprehensive DHCP answer
comprehensive_answer = """The Dynamic Host Configuration Protocol (DHCP) is a network management protocol used in Internet Protocol (IP) networks, whereby a DHCP server dynamically assigns an IP address and other network configuration parameters to each device on the network. DHCP has largely replaced RARP (and BOOTP). Uses of DHCP are: Simplifies installation and configuration of end systems, Allows for manual and automatic IP address assignment, May provide additional configuration information (DNS server, netmask, default router, etc.)"""

print("=" * 70)
print("TEST 1: Comprehensive DHCP Answer (should score high)")
print("=" * 70)
r1 = requests.post('http://localhost:8000/predict', json={
    'question': 'What is the Dynamic Host Configuration Protocol (DHCP)? What is it used for?',
    'answer': comprehensive_answer
})
print(json.dumps(r1.json(), indent=2))

print("\n" + "=" * 70)
print("TEST 2: Partial DHCP Answer (should score medium)")
print("=" * 70)
r2 = requests.post('http://localhost:8000/predict', json={
    'question': 'What is the Dynamic Host Configuration Protocol (DHCP)? What is it used for?',
    'answer': 'DHCP is a protocol that assigns IP addresses automatically to devices on a network.'
})
print(json.dumps(r2.json(), indent=2))

print("\n" + "=" * 70)
print("TEST 3: Off-topic Answer (should score 0)")
print("=" * 70)
r3 = requests.post('http://localhost:8000/predict', json={
    'question': 'What is the Dynamic Host Configuration Protocol (DHCP)? What is it used for?',
    'answer': 'AI, or Artificial Intelligence, is a field of computer science.'
})
print(json.dumps(r3.json(), indent=2))
