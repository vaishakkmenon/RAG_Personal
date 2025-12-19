@echo off
REM Docker Test Runner Script for Windows
REM Usage: run_docker_tests.bat [options]

REM Load API_KEY from .env file
for /f "tokens=1,2 delims==" %%a in ('type .env ^| findstr /i "^API_KEY="') do set %%a=%%b

REM Default values
if "%API_KEY%"=="" set API_KEY=dev-key-1

echo ================================================================================
echo Running RAG Tests in Docker
echo ================================================================================
echo API URL: http://api:8000
echo API Key: %API_KEY%
echo.

docker-compose run --rm test /opt/venv/bin/python tests/runners/run_tests.py ^
  --api-url http://api:8000 ^
  --api-key %API_KEY% ^
  --answers-file /workspace/data/eval/test_answers.json ^
  --report-file /workspace/data/eval/test_validation_report.json ^
  %*

echo.
echo ================================================================================
echo Tests Complete
echo ================================================================================
