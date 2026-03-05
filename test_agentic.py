# test_agent.py - Quick test for agentic capabilities
from agentic_rag import UrRightAgent

print("🤖 Testing UrRight Agentic RAG...")
agent = UrRightAgent()

# Test queries showing different capabilities
test_queries = [
    ("simple", "What are my rights if arrested?"),
    ("swahili", "Na haki zangu nikamatwa na polisi?"),
    ("complex", "If police arrest me illegally, what can I do?"),
]

for test_type, query in test_queries:
    print(f"\n{'-'*50}")
    print(f"Testing {test_type} query: {query}")
    print(f"{'-'*50}")
    
    result = agent.process_query(query, "test_session")
    print(f"\nResponse: {result['response']}")
    if result['sources']:
        print(f"\nSources: {result['sources']}")
    
    input("\nPress Enter for next test...")