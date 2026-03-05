-- init.sql - Safe database initialization
-- Database already created by POSTGRES_DB environment variable
-- Just grant privileges (safe to run even if already granted)

-- Grant all privileges on the existing database
GRANT ALL PRIVILEGES ON DATABASE urright_db TO urright_user;

-- Connect to the database
\c urright_db;

-- Create tables if they don't exist (optional - your app will create them)
-- This just ensures permissions are set correctly
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO urright_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO urright_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO urright_user;