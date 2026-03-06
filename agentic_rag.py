# agentic_rag.py - Agentic RAG for Kenyan Constitution
# PRODUCTION GRADE - Pure AI Reasoning with Enhanced Safety & Accuracy

import os
import logging
import asyncio
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings
from groq import Groq

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# Configuration
# ============================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.error("Please set GROQ_API_KEY in .env file")
    exit(1)

DEFAULT_MODEL = "llama-3.3-70b-versatile"
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize ChromaDB
try:
    chroma_client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "localhost"),
        port=int(os.getenv("CHROMA_PORT", "8000")),
        settings=Settings(allow_reset=True, anonymized_telemetry=False)
    )
    chroma_client.heartbeat()
    logger.info("Connected to ChromaDB server")
except:
    logger.info("Using local ChromaDB storage")
    chroma_client = chromadb.PersistentClient(
        path="./chroma_data",
        settings=Settings(allow_reset=True, anonymized_telemetry=False)
    )

COLLECTION_NAME = "kenyan_constitution"

# ============================================
# KENYAN LEGAL FRAMEWORK - For Awareness Only (Not Hardcoded)
# ============================================

KENYAN_LAWS = {
    # Land & Property
    "Land Act, 2012": "Governs public land, compulsory acquisition, and leases.",
    "Land Registration Act, 2012": "Consolidates land registration systems.",
    "National Land Commission Act, 2012": "Establishes the NLC to manage public land.",
    "Community Land Act, 2016": "Secures collective land rights for communities.",
    "Land Control Act (Cap 302)": "Regulates transactions in agricultural land.",
    "Physical and Land Use Planning Act, 2019": "Regulates land use planning and development.",
    "Sectional Properties Act, 2020": "Governs ownership of sectional units (apartments).",
    "Law of Succession Act (Cap 160)": "Governs inheritance and wills.",
    
    # Environment
    "Environmental Management and Co-ordination Act (EMCA), 1999": "Governs environmental protection.",
    "Forest Conservation and Management Act, 2016": "Regulates forest conservation.",
    "Water Act, 2016": "Governs water resources management.",
    "Wildlife Conservation and Management Act, 2013": "Protects wildlife and habitats.",
    
    # Governance
    "County Governments Act, 2012": "Implements devolved government.",
    "Intergovernmental Relations Act, 2012": "Governs relations between national and county governments.",
    "Public Finance Management Act, 2012": "Regulates public financial management.",
    "Leadership and Integrity Act, 2012": "Implements Chapter 6 of the Constitution on leadership.",
    
    # Judicial
    "Environment and Land Court Act, 2011": "Establishes the specialized court for land disputes.",
    "Judicature Act (Cap 8)": "Governs the application of common law and doctrines of equity.",
    "Civil Procedure Act (Cap 21)": "Regulates civil court procedures.",
    
    # Family & Children
    "Marriage Act, 2014": "Consolidates all marriage laws.",
    "Children Act, 2022": "Protects children's rights and welfare.",
    "Protection Against Domestic Violence Act, 2015": "Protects victims of domestic violence.",
    "Sexual Offences Act, 2006": "Criminalizes sexual offenses and protects victims.",
    
    # Employment
    "Employment Act, 2007": "Governs employment rights and relations.",
    "Labour Relations Act, 2007": "Regulates trade unions and collective bargaining.",
    "Work Injury Benefits Act, 2007": "Provides compensation for workplace injuries.",
    
    # Education
    "Basic Education Act, 2013": "Governs primary and secondary education.",
    "Teachers Service Commission Act, 2012": "Regulates teacher conduct and discipline.",
    "Universities Act, 2012": "Governs university education.",
    
    # Health
    "Health Act, 2017": "Governs health services and systems.",
    "Mental Health Act, 1989": "Regulates mental health services.",
    
    # Consumer & Business
    "Companies Act, 2015": "Governs corporate entities and business registration.",
    "Consumer Protection Act, 2012": "Protects consumer rights.",
    "Competition Act, 2010": "Regulates competition and fair trade.",
    
    # Criminal
    "Penal Code (Cap 63)": "Defines criminal offenses and penalties.",
    "Criminal Procedure Code (Cap 75)": "Regulates criminal court procedures.",
    
    # Traffic & Transport
    "Traffic Act (Cap 403)": "Regulates road traffic and offenses.",
    "National Transport and Safety Authority Act, 2012": "Establishes NTSA for road safety.",
    
    # Tax
    "Income Tax Act (Cap 470)": "Governs income tax collection.",
    "Value Added Tax Act, 2013": "Regulates VAT collection.",
    "Tax Procedures Act, 2015": "Sets tax administration procedures.",
    "Kenya Revenue Authority Act, 1995": "Establishes KRA for tax collection.",
    
    # Professional Regulation
    "Engineers Act, 2011": "Regulates engineering profession.",
    "Architects and Quantity Surveyors Act (Cap 525)": "Regulates architecture profession.",
    "Medical Practitioners and Dentists Act (Cap 253)": "Regulates medical professionals.",
    "Advocates Act (Cap 16)": "Regulates legal profession."
}

# Child protection resources
CHILD_HELPLINES = {
    "childline": "116 (toll-free, 24/7)",
    "children_office": "Contact your nearest County Children's Office",
    "police": "Police station - ask for Child Protection Unit",
    "tkd": "Tumaini La Dada (gender-based violence): 0800720545"
}

# ============================================
# System Prompt - STRONG KENYAN IDENTITY with Statute Awareness
# ============================================

SYSTEM_PROMPT = f"""
You are UrRight, an AI assistant specialized SOLELY in the Kenyan Constitution (2010) and Kenyan laws.

**🇰🇪 ABSOLUTE IDENTITY RULES - NEVER FORGET:**
- You are KENYAN. You NEVER ask for country/state - users are ALWAYS in Kenya
- You ALWAYS refer to Kenyan institutions (KRA, County Governments, Courts, IPOA, NTSA, TSC, Children's Office)
- You ALWAYS cite Kenyan laws - you KNOW these laws exist but NOT their full text
- You NEVER give generic advice - EVERY answer must be Kenya-specific
- If a matter falls under a specific Act (like Succession Act, Children's Act), you ACKNOWLEDGE that and guide appropriately

**📚 KENYAN LAWS YOU SHOULD KNOW EXIST (for awareness only):**
{chr(10).join([f"- {law}: {desc}" for law, desc in list(KENYAN_LAWS.items())[:15]])}...

**🛡️ CHILD SAFEGUARDING PROTOCOL - MANDATORY:**
If the user might be a child (mentions school, teacher, parent, age under 18):
1. Use SIMPLER language
2. Include helpline: Childline Kenya 116 (toll-free, 24/7)
3. ENCOURAGE telling a trusted adult
4. NEVER pressure for identifying details
5. Be EXTRA gentle and supportive

**⚠️ URGENCY DETECTION:**
If situation involves: violence, abuse, arrest, threat, emergency:
- Acknowledge urgency FIRST
- Suggest immediate actions
- Provide emergency contacts
- Offer to guide step-by-step

**🏛️ CORRECT INSTITUTION REFERENCES (BE ACCURATE):**
- PROBATE/INHERITANCE → High Court (NOT County Government)
- LAND DISPUTES → Environment and Land Court OR NLC
- TEACHER MISCONDUCT → TSC + School Board
- CHILD ABUSE → Children's Office + Police + Childline 116
- TRAFFIC OFFENSES → NTSA + Police
- TAX MATTERS → KRA (NOT County for income tax)
- COUNTY TAXES → County Government (for rates, fees)

**YOUR TONE:**
- Warm, friendly, encouraging
- Use simple language
- Be patient and empathetic

**MEMORY & REASONING RULES:**
- REMEMBER what you asked and what user told you
- NEVER repeat questions
- BUILD on previous answers
- TRACK gathered details

**LANGUAGE RULES:**
- If user writes in English → Respond ONLY in English
- If user writes in Swahili → Respond ONLY in Swahili
- NEVER mix languages

**THE INTELLIGENT APPROACH:**

When a user describes a PERSONAL SITUATION:
1. UNDERSTAND what they're facing
2. EMPATHIZE first
3. IDENTIFY missing critical info
4. ASK intelligent questions
5. After details, provide PERSONALIZED guidance

**RESPONSE STRUCTURE:**
- Personal situation (first response): Empathy + need more info + 2-4 questions
- After details: Thank you + reference relevant laws + practical steps + offer more help
- General questions: Clear explanation + laws + practical steps + disclaimer

**⚠️ HONESTY ABOUT LIMITATIONS:**
If a situation falls under a specific Act you don't have full text of:
- Name the Act
- Explain general principles
- Guide to the right authority
- Suggest consulting a lawyer or visiting kenyalaw.org

**ALWAYS end with:**
--- 
*This response is AI-generated. For accurate guidance, consult a lawyer or legal institution.*

You are Kenya's constitutional assistant. Own it. Be proud. Help every Kenyan know their rights. 🇰🇪
"""

# ============================================
# Constitution Tool (Improved - Better article extraction)
# ============================================

class ConstitutionTool:
    """Tool for searching the Kenyan Constitution"""
    
    # Articles that are commonly misused - we'll filter these out based on context
    IRRELEVANT_ARTICLES_FOR_CONTEXT = {
        "general": [],  # No filter for general
        "inheritance": ["133", "140", "144", "147", "159", "160", "166", "169", "183"],
        "abuse": ["133", "140", "144", "147", "159", "160", "166", "169", "183"],
        "land": ["133", "140", "144", "147", "159", "160", "166", "169"],
        "traffic": ["133", "140", "144", "147", "159", "160", "166", "169"],
        "employment": ["133", "140", "144", "147", "159", "160", "166", "169"],
    }
    
    def __init__(self, collection_name=COLLECTION_NAME):
        self.collection_name = collection_name
        self.collection = None
        self._init_collection()
    
    def _init_collection(self):
        try:
            self.collection = chroma_client.get_collection(self.collection_name)
            logger.info(f"✅ Connected to collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"❌ Could not connect to collection: {e}")
    
    def _extract_articles_from_text(self, text: str) -> List[str]:
        """Extract article numbers from text for better source accuracy"""
        articles = set()
        
        patterns = [
            r'[Aa]rticle\s+(\d+)',
            r'[Aa]RTICLE\s+(\d+)',
            r'[Ss]ection\s+(\d+)',
            r'[Kk]ifungu\s+cha\s+(\d+)',
            r'[Ii]bara\s+(\d+)',
            r'§\s*(\d+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                articles.add(f"Article {match}")
        
        return list(articles)
    
    def filter_articles_by_context(self, articles: List[str], context_type: str = "general") -> List[str]:
        """Filter out articles that are clearly irrelevant to the context"""
        if not articles or context_type not in self.IRRELEVANT_ARTICLES_FOR_CONTEXT:
            return articles
        
        irrelevant = self.IRRELEVANT_ARTICLES_FOR_CONTEXT[context_type]
        filtered = []
        
        for article in articles:
            # Extract article number
            match = re.search(r'(\d+)', article)
            if match:
                article_num = match.group(1)
                if article_num not in irrelevant:
                    filtered.append(article)
            else:
                # Keep if no number found
                filtered.append(article)
        
        return filtered if filtered else articles
    
    def search(self, query: str, n_results: int = 5, context_type: str = "general") -> Dict:
        if not self.collection:
            return {"error": "Constitution not indexed", "results": []}
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)
            formatted_results = []
            if results and results['documents'] and results['documents'][0]:
                for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                    # Extract articles from the document text
                    doc_articles = self._extract_articles_from_text(doc)
                    metadata_article = metadata.get('article', 'Unknown')
                    
                    # Combine all articles found
                    all_articles = []
                    if metadata_article and metadata_article != "Unknown":
                        all_articles.append(metadata_article)
                    all_articles.extend(doc_articles)
                    
                    # Remove duplicates
                    unique_articles = list(set(all_articles))
                    
                    # Filter by context
                    filtered_articles = self.filter_articles_by_context(unique_articles, context_type)
                    
                    formatted_results.append({
                        "full_content": doc,
                        "article": metadata_article,
                        "all_articles": filtered_articles
                    })
            return {"results": formatted_results}
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {"error": str(e), "results": []}

# ============================================
# Translation Tool (Enhanced)
# ============================================

class TranslationTool:
    """Language detection - lets AI do the heavy lifting"""
    
    def detect_language(self, text: str) -> str:
        """Basic detection - AI will understand context better"""
        text_lower = text.lower()
        
        swahili_indicators = [
            'haki', 'na', 'ya', 'kwa', 'ni', 'watu', 'kama', 'hii', 'sheria',
            'katiba', 'serikali', 'rais', 'bunge', 'mahakama', 'polisi',
            'nini', 'kwanini', 'vipi', 'lini', 'wapi', 'nani', 'gani',
            'habari', 'jambo', 'asante', 'tafadhali', 'tuko', 'yako', 'zako',
            'wako', 'yetu', 'etu', 'hizo', 'hizi', 'hilo', 'hili'
        ]
        
        words = text_lower.split()
        swahili_count = sum(1 for word in words if word in swahili_indicators)
        
        return "swahili" if swahili_count >= 2 else "english"

# ============================================
# Conversation Memory - ENHANCED for better tracking
# ============================================

class ConversationMemory:
    """Stores conversation history with enhanced tracking"""
    
    def __init__(self, max_history=15):
        self.max_history = max_history
        self.memories = {}
        self.awaiting_details = {}
        self.collected_details = {}
        self.conversation_stage = {}
        self.questions_asked = {}
        self.user_answers = {}
        self.last_topic = {}
        self.urgency_flags = {}  # Track urgency of situation
        self.child_safeguarding = {}  # Flag if user might be a child
    
    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to memory"""
        if session_id not in self.memories:
            self._init_session(session_id)
        
        self.memories[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        
        # Check if user might be a child (simple heuristic)
        if role == "user" and self._might_be_child(content):
            self.child_safeguarding[session_id] = True
        
        # If user message and awaiting details, store as answer
        if role == "user" and self.awaiting_details.get(session_id, False):
            self.user_answers[session_id].append(content)
        
        # Keep history manageable
        if len(self.memories[session_id]) > self.max_history:
            self.memories[session_id] = self.memories[session_id][-self.max_history:]
    
    def _init_session(self, session_id: str):
        """Initialize a new session"""
        self.memories[session_id] = []
        self.collected_details[session_id] = []
        self.awaiting_details[session_id] = False
        self.conversation_stage[session_id] = "initial"
        self.questions_asked[session_id] = []
        self.user_answers[session_id] = []
        self.last_topic[session_id] = None
        self.urgency_flags[session_id] = "low"
        self.child_safeguarding[session_id] = False
    
    def _might_be_child(self, text: str) -> bool:
        """Simple heuristic to detect if user might be a child"""
        text_lower = text.lower()
        child_indicators = [
            'my teacher', 'at school', 'in class', 'my parent', 'my mom', 'my dad',
            'i am 1', 'i am 2', 'i am 3', 'i am 4', 'i am 5', 'i am 6', 'i am 7',
            'i am 8', 'i am 9', 'i am 10', 'i am 11', 'i am 12', 'i am 13',
            'i am 14', 'i am 15', 'i am 16', 'i am 17', 'years old'
        ]
        return any(indicator in text_lower for indicator in child_indicators)
    
    def add_question(self, session_id: str, question: str):
        """Record a question asked by the assistant"""
        if session_id not in self.questions_asked:
            self.questions_asked[session_id] = []
        self.questions_asked[session_id].append(question)
    
    def get_asked_questions(self, session_id: str) -> List[str]:
        """Get all questions already asked in this session"""
        return self.questions_asked.get(session_id, [])
    
    def get_context(self, session_id: str) -> str:
        """Get enhanced conversation history for context"""
        if session_id not in self.memories:
            return ""
        
        context = "CONVERSATION HISTORY:\n"
        
        # Last 5 messages
        for msg in self.memories[session_id][-5:]:
            role = "USER" if msg['role'] == 'user' else "ASSISTANT"
            context += f"{role}: {msg['content']}\n"
        
        # Add summary of what we know
        if session_id in self.user_answers and self.user_answers[session_id]:
            context += f"\nUSER HAS TOLD ME: {', '.join(self.user_answers[session_id][-3:])}\n"
        
        # Add what we've already asked
        if session_id in self.questions_asked and self.questions_asked[session_id]:
            context += f"\nI'VE ALREADY ASKED ABOUT: {', '.join(self.questions_asked[session_id][-3:])}\n"
        
        # Add flags
        if self.child_safeguarding.get(session_id):
            context += "\n⚠️ USER MAY BE A CHILD - Use appropriate language and include helplines\n"
        
        return context
    
    def set_urgency(self, session_id: str, level: str):
        """Set urgency level for session"""
        self.urgency_flags[session_id] = level
    
    def get_urgency(self, session_id: str) -> str:
        """Get urgency level"""
        return self.urgency_flags.get(session_id, "low")
    
    def is_child_session(self, session_id: str) -> bool:
        """Check if session might involve a child"""
        return self.child_safeguarding.get(session_id, False)
    
    def set_awaiting_details(self, session_id: str):
        """Mark that we're waiting for details"""
        self.awaiting_details[session_id] = True
        self.conversation_stage[session_id] = "gathering"
    
    def add_detail(self, session_id: str, detail: str):
        """Store a detail provided by user"""
        if session_id not in self.collected_details:
            self.collected_details[session_id] = []
        self.collected_details[session_id].append(detail)
    
    def get_collected_details(self, session_id: str) -> List[str]:
        """Get all details collected"""
        return self.collected_details.get(session_id, [])
    
    def is_awaiting_details(self, session_id: str) -> bool:
        """Check if we're waiting for details"""
        return self.awaiting_details.get(session_id, False)
    
    def get_conversation_stage(self, session_id: str) -> str:
        """Get current stage: initial, gathering, responding"""
        return self.conversation_stage.get(session_id, "initial")
    
    def set_last_topic(self, session_id: str, topic: str):
        """Set the last topic discussed"""
        self.last_topic[session_id] = topic
    
    def get_last_topic(self, session_id: str) -> Optional[str]:
        """Get the last topic discussed"""
        return self.last_topic.get(session_id)
    
    def clear_awaiting(self, session_id: str):
        """Clear waiting state after providing response"""
        self.awaiting_details[session_id] = False
        self.conversation_stage[session_id] = "responding"

    def get_all_sessions(self) -> List[str]:
        """Get all active session IDs"""
        return list(self.memories.keys())    
    
    def clear_session(self, session_id: str):
        """Clear all memory for a session"""
        keys = ['memories', 'awaiting_details', 'collected_details', 
                'conversation_stage', 'questions_asked', 'user_answers', 
                'last_topic', 'urgency_flags', 'child_safeguarding']
        for key in keys:
            attr = getattr(self, key)
            if session_id in attr:
                del attr[session_id]

# Global memory
conversation_memory = ConversationMemory()

# ============================================
# Source Formatter - Clean article display with context filtering
# ============================================

def format_sources(sources: List[str], context_type: str = "general") -> List[str]:
    """Clean and format source articles for display with context filtering"""
    if not sources:
        return []
    
    # Remove Unknown, None, empty
    clean = [s for s in sources if s and s != "Unknown" and s != "None" and s.strip()]
    
    # Articles that are commonly irrelevant by context
    irrelevant_map = {
        "inheritance": ["133", "140", "144", "147", "159", "160", "166", "169", "183"],
        "abuse": ["133", "140", "144", "147", "159", "160", "166", "169", "183"],
        "land": ["133", "140", "144", "147", "159", "160", "166", "169"],
        "traffic": ["133", "140", "144", "147", "159", "160", "166", "169"],
        "employment": ["133", "140", "144", "147", "159", "160", "166", "169"],
    }
    
    irrelevant = irrelevant_map.get(context_type, [])
    
    # Filter out irrelevant articles
    filtered = []
    for s in clean:
        match = re.search(r'(\d+)', s)
        if match:
            if match.group(1) not in irrelevant:
                filtered.append(s)
        else:
            filtered.append(s)
    
    # Extract article numbers for sorting
    numbered = []
    others = []
    
    for s in filtered:
        match = re.search(r'(\d+)', s)
        if match:
            try:
                num = int(match.group(1))
                numbered.append((num, s))
            except:
                others.append(s)
        else:
            others.append(s)
    
    # Sort numbered articles
    numbered.sort()
    result = [s[1] for s in numbered]
    
    # Add non-numbered at the end
    result.extend(sorted(others))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_result = []
    for item in result:
        if item not in seen:
            seen.add(item)
            unique_result.append(item)
    
    return unique_result

# ============================================
# Child Safeguarding Helper
# ============================================

def get_child_safeguarding_message(language: str) -> str:
    """Get child safeguarding message with helplines"""
    if language == "swahili":
        return (
            "\n\n**🛡️ UNATILIWA KUSAIDIWA**\n"
            "• Piga simu **116** (Childline Kenya - bure, 24/7)\n"
            "• Mwambie mwalimu, mzazi, au mtu mzima unayemwamini\n"
            "• Wasiliana na Ofisi ya Watoto karibu nawe\n"
            "• Una haki ya kuwa salama - hii si kosa lako!"
        )
    return (
        "\n\n**🛡️ YOU DESERVE HELP**\n"
        "• Call **116** (Childline Kenya - toll-free, 24/7)\n"
        "• Tell a teacher, parent, or trusted adult\n"
        "• Visit your nearest Children's Office\n"
        "• You have the right to be safe - this is NOT your fault!"
    )

def get_urgency_message(urgency: str, language: str) -> str:
    """Get urgency-aware message"""
    if urgency == "high":
        if language == "swahili":
            return "\n\n**⚠️ HII NI DHARURA - Chukua hatua sasa hivi!**"
        return "\n\n**⚠️ THIS IS URGENT - Take action now!**"
    return ""

# ============================================
# The INTELLIGENT Agent - No Hardcoding
# ============================================

class UrRightAgent:
    """Pure AI agent - No scripted topics, no pre-written questions"""
    
    def __init__(self, model=DEFAULT_MODEL):
        self.model = model
        self.constitution_tool = ConstitutionTool()
        self.translation_tool = TranslationTool()
        self.system_prompt = SYSTEM_PROMPT
        logger.info(f"✅ UrRight Agent initialized with model: {model}")
    
    def _detect_context_type(self, analysis: Dict) -> str:
        """Detect context type for article filtering"""
        situation = analysis.get("situation_type", "").lower()
        if any(word in situation for word in ["inherit", "will", "estate", "succession"]):
            return "inheritance"
        elif any(word in situation for word in ["abuse", "teacher", "child", "bully"]):
            return "abuse"
        elif any(word in situation for word in ["land", "property", "shamba"]):
            return "land"
        elif any(word in situation for word in ["traffic", "driving", "speeding"]):
            return "traffic"
        elif any(word in situation for word in ["job", "work", "employ", "salary"]):
            return "employment"
        return "general"
    
    # ===== INTELLIGENT SITUATION ANALYZER =====
    def _analyze_situation(self, query: str, context: str, language: str, session_id: str = None) -> Dict:
        """
        Enhanced analysis with memory of previous interactions and statute awareness
        """
        asked = conversation_memory.get_asked_questions(session_id) if session_id else []
        answers = conversation_memory.user_answers.get(session_id, []) if session_id else []
        is_child = conversation_memory.is_child_session(session_id) if session_id else False
        
        # Add Kenyan laws list for awareness (truncated for token limits)
        laws_sample = list(KENYAN_LAWS.keys())[:20]
        
        prompt = f"""
You are an intelligent Kenyan legal assistant. Analyze this user's message:

USER MESSAGE: "{query}"

CONVERSATION SO FAR:
{context}

QUESTIONS I'VE ASKED: {', '.join(asked[-3:]) if asked else 'None'}
USER HAS TOLD ME: {', '.join(answers[-3:]) if answers else 'Nothing yet'}

KNOWN KENYAN LAWS (for awareness):
{', '.join(laws_sample)}...

Based on ALL information:

1. What TYPE of situation? (be specific - e.g., "teacher emotional abuse", "inheritance dispute")
2. PERSONAL situation or GENERAL question?
3. What details do you ALREADY have?
4. What CRITICAL information is STILL MISSING? (DO NOT ask for what they've already told you)
5. Which KENYAN LAW primarily governs this? (Name the specific Act if applicable)
6. Is this in the CONSTITUTION or another law?
7. Which Kenyan institutions are responsible? (BE ACCURATE - courts not county for probate, etc.)
8. What is the URGENCY level? (high if violence/abuse/arrest, medium/low otherwise)

Respond in JSON ONLY:
{{
    "situation_type": "detailed description",
    "is_personal": true/false,
    "provided_details": ["list", "of", "details", "already", "known"],
    "missing_info": ["what", "STILL", "needed"],
    "primary_law": "Name of Act or 'Constitution'",
    "law_category": "Constitution/Land/Family/Criminal/etc",
    "responsible_institutions": ["institution1", "institution2"],
    "urgency": "low/medium/high",
    "needs_followup": true/false
}}
"""
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=700
            )
            result = json.loads(response.choices[0].message.content)
            
            # Store urgency in memory
            if session_id:
                conversation_memory.set_urgency(session_id, result.get("urgency", "low"))
            
            logger.info(f"🧠 Analysis: {result['situation_type']} | Law: {result.get('primary_law', 'Unknown')}")
            return result
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {
                "situation_type": "unknown",
                "is_personal": False,
                "provided_details": [],
                "missing_info": [],
                "primary_law": "Unknown",
                "law_category": "general",
                "responsible_institutions": [],
                "urgency": "low",
                "needs_followup": False
            }
    
    # ===== INTELLIGENT QUESTION GENERATOR =====
    def _generate_followup_questions(self, analysis: Dict, language: str, session_id: str = None) -> str:
        """
        Generate questions based on missing info, avoiding repetition
        """
        missing = analysis.get("missing_info", [])
        situation = analysis.get("situation_type", "this situation")
        institutions = analysis.get("responsible_institutions", [])
        is_child = conversation_memory.is_child_session(session_id) if session_id else False
        
        if not missing:
            return ""
        
        asked = conversation_memory.get_asked_questions(session_id) if session_id else []
        
        # Adjust language for child users
        tone = "Use VERY simple language suitable for a child. Be extra gentle and supportive." if is_child else "Be warm and professional."
        
        prompt = f"""
The user is facing: {situation}
Already known: {', '.join(analysis.get('provided_details', ['nothing']))}
Need to know: {', '.join(missing)}
Relevant authorities: {', '.join(institutions) if institutions else 'General'}

Already asked (DO NOT REPEAT): {', '.join(asked[-3:]) if asked else 'None'}

{tone}

Generate 2-3 follow-up questions in {language.upper()} that:
1. Are SPECIFIC to THEIR Kenyan situation
2. Gather ONLY missing information
3. Reference correct Kenyan institutions
4. Are warm and empathetic

Return ONLY numbered questions, nothing else.
"""
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            questions = response.choices[0].message.content
            
            if session_id:
                conversation_memory.add_question(session_id, questions)
            
            return questions
        except Exception as e:
            logger.error(f"Question generation error: {e}")
            fallback = "1. Can you tell me more about what happened?\n2. When did this happen?\n3. Do you have any documents?"
            return fallback
    
    # ===== INTELLIGENT RESPONSE GENERATOR =====
    def _generate_personalized_response(self, query: str, details: List[str], 
                                        analysis: Dict, chunks: List[str], 
                                        language: str, session_id: str = None) -> str:
        """
        Generate truly personalized Kenyan-specific response with safeguards
        """
        chunks_text = "\n".join([c[:500] for c in chunks[:3]]) if chunks else "No specific articles found."
        details_text = "\n".join([f"- {d}" for d in details]) if details else "No additional details provided."
        institutions = analysis.get("responsible_institutions", [])
        law_name = analysis.get("primary_law", "Kenyan law")
        law_category = analysis.get("law_category", "general")
        is_child = conversation_memory.is_child_session(session_id) if session_id else False
        urgency = conversation_memory.get_urgency(session_id) if session_id else "low"
        
        # Add urgency message if needed
        urgency_msg = get_urgency_message(urgency, language)
        
        # Add child safeguarding if needed
        child_msg = get_child_safeguarding_message(language) if is_child else ""
        
        # Different guidance based on whether we have the full law
        if law_category == "Constitution" and chunks:
            law_guidance = "Based on the Constitution, here's what you should know:"
        else:
            law_guidance = f"This falls under the **{law_name}**. While I don't have the full text, I can explain the general process and guide you to the right authority."
        
        prompt = f"""
You are UrRight, a Kenyan legal assistant.

SITUATION: {analysis.get('situation_type', 'Unknown')}
USER: "{query}"
DETAILS: {details_text}

LAW: {law_name}
AUTHORITIES: {', '.join(institutions) if institutions else 'General'}

CONSTITUTIONAL TEXT:
{chunks_text}

{law_guidance}

TASK: Provide PERSONALIZED Kenyan-specific guidance that:
1. ACKNOWLEDGES their situation with empathy
2. EXPLAINS in SIMPLE language
3. GIVES practical steps tailored to THEIR details
4. MENTIONS correct Kenyan institutions
5. OFFERS more help

{urgency_msg}
{child_msg}

Be warm, specific, helpful. Use {language.upper()} only.

Remember the disclaimer at the end separated by "---".
"""
        try:
            response = groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1300
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return f"I encountered an error. Please try again. {self._get_disclaimer(language)}"
    
    # ===== INTELLIGENT GREETING DETECTION =====
    def _check_greeting(self, query: str) -> Optional[str]:
        """
        Intelligent greeting detection - uses AI to distinguish between greetings and questions
        """
        q = query.lower().strip()
        
        # Common greeting patterns (exact matches or very short)
        simple_greetings = {
            'hello', 'hi', 'hey', 'howdy', 'greetings',
            'habari', 'jambo', 'mambo', 'vipi', 'hujambo', 'shikamoo', 'sasa',
            'good morning', 'good afternoon', 'good evening',
            'morning', 'afternoon', 'evening'
        }
        
        # Check for exact matches or very short queries (likely greetings)
        if q in simple_greetings or (len(q.split()) <= 2 and any(g in q for g in simple_greetings)):
            # Check if it's actually a greeting or just a word
            if q.startswith(('hello', 'hi', 'hey', 'habari', 'jambo')):
                return "🇰🇪 👋 Hello! I'm UrRight, your Kenyan constitutional assistant. How can I help you understand your rights today?"
        
        # For longer queries, let the AI decide
        if len(q.split()) > 3:
            prompt = f"""
Determine if the following user message is JUST a greeting/small talk or a REAL question about Kenyan law.

User message: "{query}"

If it's JUST a greeting/small talk (like hello, hi, how are you, etc.), respond with "GREETING".
If it's asking a REAL question about law, rights, constitution, or any Kenyan matter, respond with "QUESTION".

Respond with ONLY one word: GREETING or QUESTION
"""
            try:
                response = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=10
                )
                result = response.choices[0].message.content.strip().upper()
                
                if "GREETING" in result:
                    # Check language for appropriate greeting response
                    if any(word in q for word in ['habari', 'jambo', 'mambo', 'vipi', 'hujambo', 'shikamoo']):
                        return "🇰🇪 👋 Habari! Mimi ni UrRight, msaidizi wako wa katiba ya Kenya. Ninaweza kukusaidia vipi leo?"
                    else:
                        return "🇰🇪 👋 Hello! I'm UrRight, your Kenyan constitutional assistant. How can I help you understand your rights today?"
            except:
                # If AI fails, fall back to simple detection
                pass
        
        # Check for thanks/goodbye (simpler patterns)
        thanks_words = ['thank', 'thanks', 'asante', 'thankyou']
        bye_words = ['bye', 'goodbye', 'kwaheri', 'see you', 'exit', 'quit']
        
        if any(t in q for t in thanks_words) and len(q.split()) < 5:
            return "You're most welcome! 😊 Is there anything else about your rights I can help with?"
        
        if any(b in q for b in bye_words) and len(q.split()) < 5:
            return "Goodbye! Stay informed and know your rights. Come back anytime! 🇰🇪"
        
        return None
    
    def _get_disclaimer(self, language: str) -> str:
        """Get disclaimer in appropriate language"""
        if language == "swahili":
            return "\n---\n*Jibu hili limetolewa na AI. Kwa mwongozo sahihi zaidi, wasiliana na wakili au taasisi ya kisheria.*"
        return "\n---\n*This response is AI-generated. For accurate guidance, consult a lawyer or legal institution.*"
    
    # ===== MAIN PROCESSING =====
    def process_query(self, query: str, session_id: str = None, requested_language: str = None) -> Dict:
        """
        Production-grade processing with all fixes implemented
        """
        if not session_id:
            session_id = f"session_{datetime.now().timestamp()}"
        
        # Language detection
        if requested_language:
            lang_map = {'en': 'english', 'sw': 'swahili'}
            language = lang_map.get(requested_language, 'english')
        else:
            language = self.translation_tool.detect_language(query)
        
        logger.info(f"🔍 Language: {language} | Session: {session_id}")
        
        # Check for greetings
        greeting = self._check_greeting(query)
        if greeting:
            conversation_memory.add_message(session_id, "user", query)
            conversation_memory.add_message(session_id, "assistant", greeting)
            return {
                "response": greeting,
                "sources": [],
                "session_id": session_id,
                "language": language,
                "chunks_retrieved": 0
            }
        
        # Get conversation context
        context = conversation_memory.get_context(session_id)
        
        # Check if in follow-up mode
        if conversation_memory.is_awaiting_details(session_id):
            logger.info(f"📝 Follow-up for session {session_id}")
            
            conversation_memory.add_detail(session_id, query)
            conversation_memory.add_message(session_id, "user", query)
            details = conversation_memory.get_collected_details(session_id)
            
            logger.info(f"📝 Collected {len(details)} details")
            
            full_context = context
            analysis = self._analyze_situation(
                conversation_memory.memories[session_id][0]['content'] if conversation_memory.memories[session_id] else query,
                full_context,
                language,
                session_id
            )
            
            # Get context type for filtering
            context_type = self._detect_context_type(analysis)
            
            # Search with context-aware filtering
            search_query = query + " " + " ".join(details[-3:])
            search_results = self.constitution_tool.search(search_query, n_results=5, context_type=context_type)
            
            all_sources = []
            chunks = []
            for r in search_results.get('results', []):
                chunks.append(r['full_content'])
                if r.get('all_articles'):
                    all_sources.extend(r['all_articles'])
                elif r.get('article') and r['article'] != "Unknown":
                    all_sources.append(r['article'])
            
            # Format sources with context filtering
            formatted_sources = format_sources(all_sources, context_type)
            
            # Generate response
            response = self._generate_personalized_response(
                query, details, analysis, chunks, language, session_id
            )
            
            conversation_memory.add_message(session_id, "assistant", response)
            conversation_memory.clear_awaiting(session_id)
            
            return {
                "response": response,
                "sources": formatted_sources,
                "session_id": session_id,
                "language": language,
                "chunks_retrieved": len(chunks)
            }
        
        # Analyze situation
        analysis = self._analyze_situation(query, context, language, session_id)
        conversation_memory.set_last_topic(session_id, analysis.get('situation_type', 'general'))
        
        # Set urgency
        conversation_memory.set_urgency(session_id, analysis.get('urgency', 'low'))
        
        # Personal situation needing follow-up
        if analysis.get("is_personal", False) and analysis.get("needs_followup", True):
            logger.info(f"🔍 Personal: {analysis['situation_type']}")
            
            questions = self._generate_followup_questions(analysis, language, session_id)
            
            if questions:
                # Add child helpline if needed
                child_help = ""
                if conversation_memory.is_child_session(session_id):
                    child_help = "\n\n🛡️ Remember, you can call **116** (Childline Kenya) anytime if you need to talk to someone."
                
                response = f"I understand. To help you properly, please tell me:\n\n{questions}{child_help}"
                
                conversation_memory.set_awaiting_details(session_id)
                conversation_memory.add_message(session_id, "user", query)
                conversation_memory.add_message(session_id, "assistant", response)
                
                return {
                    "response": response,
                    "sources": [],
                    "session_id": session_id,
                    "language": language,
                    "chunks_retrieved": 0,
                    "is_followup": True,
                    "awaiting_details": True
                }
        
        # Normal query
        logger.info("🔍 Searching constitution...")
        context_type = self._detect_context_type(analysis)
        search_results = self.constitution_tool.search(query, n_results=5, context_type=context_type)
        
        if "error" in search_results:
            return {
                "response": f"I'm having trouble accessing the constitution. {self._get_disclaimer(language)}",
                "sources": [],
                "session_id": session_id,
                "language": language
            }
        
        all_sources = []
        chunks = []
        for r in search_results.get('results', []):
            chunks.append(r['full_content'])
            if r.get('all_articles'):
                all_sources.extend(r['all_articles'])
            elif r.get('article') and r['article'] != "Unknown":
                all_sources.append(r['article'])
        
        formatted_sources = format_sources(all_sources, context_type)
        
        response = self._generate_personalized_response(
            query, [], analysis, chunks, language, session_id
        )
        
        conversation_memory.add_message(session_id, "user", query)
        conversation_memory.add_message(session_id, "assistant", response)
        
        return {
            "response": response,
            "sources": formatted_sources,
            "session_id": session_id,
            "language": language,
            "chunks_retrieved": len(chunks)
        }

# ============================================
# Interactive CLI
# ============================================

async def interactive_agent_chat():
    """Interactive CLI for testing"""
    print("\n" + "="*70)
    print("🇰🇪 UrRight - Kenyan Constitution Assistant")
    print("="*70)
    print("✨ PRODUCTION GRADE - All safety features enabled")
    print("✨ Article Relevance Filtering ✅")
    print("✨ Statute Awareness (40+ Kenyan Laws) ✅")
    print("✨ Child Safeguarding Protocol ✅")
    print("✨ Institution Accuracy ✅")
    print("✨ Enhanced Memory (no repeats) ✅")
    print("✨ Urgency Detection ✅")
    print("✨ Source Formatting ✅")
    print("="*70)
    
    agent = UrRightAgent()
    
    test = agent.constitution_tool.search("rights", 1)
    if "error" in test or not test.get("results"):
        print("\n⚠️ Please run app.py first to index the constitution")
        return
    
    print(f"\n✅ Ready! Model: {agent.model}")
    print("\n📝 Type your situation. 'exit' to quit\n")
    
    session = f"session_{datetime.now().timestamp()}"
    
    while True:
        try:
            q = input("\n❓ You: ").strip()
            if q.lower() in ['exit', 'quit']:
                print("\nGoodbye! Know your rights! 🇰🇪")
                break
            if not q:
                continue
            
            print("🧠 Thinking...")
            result = agent.process_query(q, session)
            
            print(f"\n🤖 UrRight:\n{result['response']}")
            
            if result.get('sources'):
                print(f"\n📚 Sources: {', '.join(result['sources'])}")
            if result.get('awaiting_details'):
                print("📝 [I'll wait for your details]")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye! 🇰🇪")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print("Sorry, an error occurred.")

if __name__ == "__main__":
    asyncio.run(interactive_agent_chat())