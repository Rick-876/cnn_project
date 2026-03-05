"""Test the calibrated scoring on port 8001."""
import requests
import json

PORT = 8001
# Test comprehensive DHCP answer
comprehensive_answer = """The Dynamic Host Configuration Protocol (DHCP) is a network management protocol used in Internet Protocol (IP) networks, whereby a DHCP server dynamically assigns an IP address and other network configuration parameters to each device on the network. DHCP has largely replaced RARP (and BOOTP). Uses of DHCP are: Simplifies installation and configuration of end systems, Allows for manual and automatic IP address assignment, May provide additional configuration information (DNS server, netmask, default router, etc.)"""

print("=" * 70)
print("TEST 1: Comprehensive DHCP Answer (should score high)")
print("=" * 70)
r1 = requests.post(f'http://localhost:{PORT}/predict', json={
    'question': 'What is the Dynamic Host Configuration Protocol (DHCP)? What is it used for?',
    'answer': comprehensive_answer
})
if r1.status_code == 200:
    print(json.dumps(r1.json(), indent=2))
else:
    print(f"Error: Status {r1.status_code}")
    print(r1.text)

print("\n" + "=" * 70)
print("TEST 2: Partial DHCP Answer (should score medium)")
print("=" * 70)
r2 = requests.post(f'http://localhost:{PORT}/predict', json={
    'question': 'What is the Dynamic Host Configuration Protocol (DHCP)? What is it used for?',
    'answer': 'DHCP is a protocol that assigns IP addresses automatically to devices on a network.'
})
if r2.status_code == 200:
    print(json.dumps(r2.json(), indent=2))
else:
    print(f"Error: Status {r2.status_code}")
    print(r2.text)

print("\n" + "=" * 70)
print("TEST 3: Off-topic Answer (should score 0)")
print("=" * 70)
r3 = requests.post(f'http://localhost:{PORT}/predict', json={
    'question': 'What is the Dynamic Host Configuration Protocol (DHCP)? What is it used for?',
    'answer': 'AI, or Artificial Intelligence, is a field of computer science.'
})
if r3.status_code == 200:
    print(json.dumps(r3.json(), indent=2))
else:
    print(f"Error: Status {r3.status_code}")
    print(r3.text)
