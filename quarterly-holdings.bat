@echo off
echo Starting Docker Compose services...
docker compose up -d

echo Waiting for services to be ready...
timeout /t 3 /nobreak >nul

echo Opening app in browser...
start http://localhost:5000

echo Project ready! Press Ctrl+C to stop services in new window.
docker compose logs -f
