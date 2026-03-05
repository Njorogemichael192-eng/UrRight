# db_init.py - Database setup for UrRight chatbot
# Following Module 2 from Gamma presentation

import os
import sys
import logging
import asyncio
import uuid
from datetime import datetime, timezone
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, DateTime, Text, inspect
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import asyncpg

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database configuration from .env file
POSTGRES_USER = os.getenv("POSTGRES_USER", "urright_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "urright_password")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "urright_db")

# Connection URLs
ADMIN_DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/postgres"
DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

# Section 2.2: Creating the Database
def create_database_if_not_exists():
    """
    Checks if the target database exists and creates it if not.
    Uses admin connection to postgres database for creation.
    """
    try:
        logger.info(f"Checking if database '{POSTGRES_DB}' exists...")
        admin_engine = create_engine(ADMIN_DATABASE_URL)
        
        with admin_engine.connect() as conn:
            # Enable autocommit for DDL statements
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            
            # Check if database exists
            result = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :dbname"),
                {"dbname": POSTGRES_DB}
            )
            
            if not result.fetchone():
                logger.info(f"Database '{POSTGRES_DB}' does not exist. Creating...")
                # Create database
                conn.execute(text(f"CREATE DATABASE \"{POSTGRES_DB}\""))
                logger.info(f"✅ Database '{POSTGRES_DB}' created successfully")
            else:
                logger.info(f"✅ Database '{POSTGRES_DB}' already exists")
        
        admin_engine.dispose()
        return True
        
    except SQLAlchemyError as e:
        logger.error(f"❌ Error creating database: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error creating database: {e}")
        return False


# Section 2.3: Creating Tables and Indexes
def create_tables():
    """
    Creates the chat_messages table with proper constraints and indexes.
    Enables UUID extension for generating unique IDs.
    """
    try:
        logger.info("Creating tables...")
        engine = create_engine(DATABASE_URL)
        
        with engine.connect() as conn:
            # Enable autocommit
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            
            # Enable UUID extension
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
            logger.info("✅ UUID extension enabled")
            
            # Drop existing table for clean setup (only in development)
            # Comment this out in production!
            conn.execute(text('DROP TABLE IF EXISTS chat_messages'))
            logger.info("✅ Dropped existing chat_messages table (development mode)")
            
            # Create chat_messages table with constraints
            create_table_sql = """
            CREATE TABLE IF NOT EXISTS chat_messages (
                message_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                session_id UUID NOT NULL,
                user_message TEXT,
                system_message TEXT,
                role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB,  -- For storing additional info like sources
                CONSTRAINT check_message_content CHECK (
                    user_message IS NOT NULL OR system_message IS NOT NULL
                )
            );
            
            -- Create indexes for better query performance
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON chat_messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_chat_messages_timestamp ON chat_messages(timestamp);
            CREATE INDEX IF NOT EXISTS idx_chat_messages_role ON chat_messages(role);
            CREATE INDEX IF NOT EXISTS idx_chat_messages_session_timestamp ON chat_messages(session_id, timestamp DESC);
            """
            
            conn.execute(text(create_table_sql))
            logger.info("✅ Chat messages table created successfully")
            
            # Verify table exists
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            logger.info(f"Tables in database: {tables}")
            
        engine.dispose()
        return True
        
    except SQLAlchemyError as e:
        logger.error(f"❌ Error creating tables: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error creating tables: {e}")
        return False


# Section 2.4: Testing the Connection
async def test_database_connection():
    """
    Tests database connection and performs CRUD operations to verify functionality.
    Uses asyncpg for asynchronous testing.
    """
    try:
        logger.info("Testing database connection...")
        
        # Connect to the database
        conn = await asyncpg.connect(
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB
        )
        
        # Test 1: Check PostgreSQL version
        version = await conn.fetchval("SELECT version()")
        logger.info(f"✅ PostgreSQL version: {version.split(',')[0]}")
        
        # Test 2: Check if table exists
        table_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'chat_messages'
            )
        """)
        
        if table_exists:
            logger.info("✅ chat_messages table exists and is accessible")
            
            # Test 3: Insert a test record
            test_session_id = uuid.uuid4()
            test_message_id = await conn.fetchval("""
                INSERT INTO chat_messages (session_id, user_message, role, timestamp)
                VALUES ($1, $2, $3, $4)
                RETURNING message_id
            """, test_session_id, "Haki yangu ni gani nikamatwa na polisi?", "user", datetime.now(timezone.utc))
            
            logger.info(f"✅ Test record inserted with ID: {test_message_id}")
            
            # Test 4: Retrieve the test record
            retrieved = await conn.fetchrow("""
                SELECT message_id, session_id, user_message, role
                FROM chat_messages 
                WHERE message_id = $1
            """, test_message_id)
            
            if retrieved:
                logger.info(f"✅ Test record retrieved: {dict(retrieved)}")
            else:
                logger.error("❌ Failed to retrieve test record")
            
            # Test 5: Insert assistant response
            assistant_message_id = await conn.fetchval("""
                INSERT INTO chat_messages (session_id, user_message, role, timestamp, metadata)
                VALUES ($1, $2, $3, $4, $5::jsonb)
                RETURNING message_id
            """, test_session_id, 
            "Kulingana na Katiba ya Kenya 2010, Kifungu cha 49: Una haki ya kufahamu sababu ya kukamatwa, kusilimana na wakili, na kupelekwa mahakamani ndani ya masaa 24.", 
            "assistant", 
            datetime.now(timezone.utc),
            '{"sources": ["Article 49"], "confidence": 0.95}')
            
            logger.info(f"✅ Assistant response inserted with ID: {assistant_message_id}")
            
            # Test 6: Count messages in session
            message_count = await conn.fetchval("""
                SELECT COUNT(*) FROM chat_messages 
                WHERE session_id = $1
            """, test_session_id)
            
            logger.info(f"✅ Session has {message_count} messages")
            
            # Test 7: Get conversation history
            conversation = await conn.fetch("""
                SELECT role, user_message, timestamp 
                FROM chat_messages 
                WHERE session_id = $1 
                ORDER BY timestamp ASC
            """, test_session_id)
            
            logger.info(f"✅ Conversation history retrieved ({len(conversation)} messages)")
            for msg in conversation:
                logger.info(f"   - {msg['role']}: {msg['user_message'][:50]}...")
            
            # Clean up test data
            await conn.execute("DELETE FROM chat_messages WHERE session_id = $1", test_session_id)
            logger.info("✅ Test data cleaned up")
            
        else:
            logger.error("❌ chat_messages table does not exist!")
            return False
        
        await conn.close()
        logger.info("✅ Database connection test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        return False


def check_postgres_running():
    """
    Helper function to check if PostgreSQL is running
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((POSTGRES_HOST, int(POSTGRES_PORT)))
    sock.close()
    
    if result == 0:
        logger.info(f"✅ PostgreSQL is running on {POSTGRES_HOST}:{POSTGRES_PORT}")
        return True
    else:
        logger.error(f"❌ Cannot connect to PostgreSQL on {POSTGRES_HOST}:{POSTGRES_PORT}")
        logger.error("   Make sure PostgreSQL is installed and running")
        logger.error("   Or we'll use Docker later to run PostgreSQL")
        return False


async def main():
    """
    Main function to run all database setup steps
    """
    print("\n" + "=" * 60)
    print("🗄️  UrRight Database Setup")
    print("=" * 60)
    
    # Step 0: Check if PostgreSQL is running
    print("\n📡 Checking PostgreSQL connection...")
    postgres_running = check_postgres_running()
    
    if not postgres_running:
        print("\n⚠️  PostgreSQL is not running locally.")
        print("   Don't worry! We have two options:")
        print("   1. Install PostgreSQL locally (for development)")
        print("   2. Use Docker (recommended - we'll do this in Module 5)")
        print("\n   For now, we'll continue with the setup.")
        print("   The database will work when we use Docker later.")
        return
    
    # Step 1: Create database if it doesn't exist
    print("\n📁 Step 1: Creating database...")
    if create_database_if_not_exists():
        print("   ✅ Database setup complete")
    else:
        print("   ❌ Database creation failed")
        return
    
    # Step 2: Create tables
    print("\n📊 Step 2: Creating tables...")
    if create_tables():
        print("   ✅ Tables created successfully")
    else:
        print("   ❌ Table creation failed")
        return
    
    # Step 3: Test connection
    print("\n🧪 Step 3: Testing database connection...")
    if await test_database_connection():
        print("   ✅ All database tests passed!")
    else:
        print("   ❌ Database tests failed")
        return
    
    print("\n" + "=" * 60)
    print("✅✅✅ UrRight Database is ready! ✅✅✅")
    print("=" * 60)
    print("\nNext step: We'll build the Naive RAG chatbot in Module 3a!")
    print("Run: python main.py (after we create it)")


if __name__ == "__main__":
    asyncio.run(main())