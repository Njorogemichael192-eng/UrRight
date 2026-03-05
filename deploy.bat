@echo off
REM deploy.bat - Deployment script for UrRight (Windows)

echo 🚀 Deploying UrRight with Docker...
echo ================================

REM Check if GROQ_API_KEY is set
if "%GROQ_API_KEY%"=="" (
    echo ❌ GROQ_API_KEY not set!
    echo Please set it with: set GROQ_API_KEY=your_key_here
    exit /b 1
)

REM Check if Constitution exists
if not exist ".\Data\kenya_constitution_2010.pdf" (
    echo ❌ Kenyan Constitution PDF not found!
    echo Please download it and place in .\Data\
    exit /b 1
)

REM Stop any running containers
echo 🛑 Stopping existing containers...
docker-compose down

REM Build and start
echo 🏗️  Building and starting containers...
docker-compose up --build -d

REM Check status
echo 📊 Container Status:
docker-compose ps

REM Show logs
echo 📝 Showing logs (Ctrl+C to exit)...
docker-compose logs -f