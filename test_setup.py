# test_setup.py
import sys
import os
from dotenv import load_dotenv

print("✅ Testing UrRight Environment Setup")
print(f"Python version: {sys.version}")

# Test .env loading
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if api_key:
    print("✅ .env file loaded successfully")
    print(f"GROQ_API_KEY exists: {api_key[:5]}...")
else:
    print("❌ Please add your GROQ_API_KEY to .env file")

# Test if constitution exists
if os.path.exists("Data/kenya_constitution_2010.pdf"):
    print("✅ Kenyan Constitution PDF found!")
else:
    print("❌ Please download the Kenyan Constitution PDF to Data folder")

print("\n🎉 Setup complete! Ready to build UrRight!")