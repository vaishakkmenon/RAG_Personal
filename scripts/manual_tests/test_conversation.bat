@echo off
REM Multi-Turn Conversation Tester for Windows
REM Requires: curl (built-in on Windows 10+)

setlocal enabledelayedexpansion

set API_URL=http://localhost:8000/chat
set API_KEY=dev-key-1

echo ================================
echo Multi-Turn Conversation Tester
echo ================================
echo.

REM Turn 1: Establish context
echo [Turn 1] What certifications do I have?
echo ----------------------------------------

curl -s -X POST "%API_URL%" ^
  -H "X-API-Key: %API_KEY%" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"What certifications do I have?\"}" > response1.json

REM Extract session_id (basic parsing without jq)
for /f "tokens=2 delims=:," %%a in ('findstr /C:"session_id" response1.json') do (
    set SESSION_ID=%%a
    set SESSION_ID=!SESSION_ID:"=!
    set SESSION_ID=!SESSION_ID: =!
    goto :found_session
)
:found_session

echo Session ID: %SESSION_ID%
echo.
timeout /t 2 /nobreak > nul

REM Turn 2: Short follow-up
echo [Turn 2] Expiration?
echo ----------------------------------------

curl -s -X POST "%API_URL%" ^
  -H "X-API-Key: %API_KEY%" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"Expiration?\", \"session_id\": \"%SESSION_ID%\"}" > response2.json

findstr /C:"grounded" response2.json | findstr /C:"true" > nul
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Short follow-up was grounded
) else (
    echo [FAIL] Short follow-up was not grounded
)

echo.
timeout /t 2 /nobreak > nul

REM Turn 3: Pronoun reference
echo [Turn 3] Which one was earned most recently?
echo ----------------------------------------

curl -s -X POST "%API_URL%" ^
  -H "X-API-Key: %API_KEY%" ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"Which one was earned most recently?\", \"session_id\": \"%SESSION_ID%\"}" > response3.json

findstr /C:"grounded" response3.json | findstr /C:"true" > nul
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] Pronoun reference was resolved
) else (
    echo [FAIL] Pronoun reference was not grounded
)

echo.
echo ================================
echo Test Complete
echo ================================
echo.
echo View full responses:
echo   response1.json, response2.json, response3.json

REM Cleanup
timeout /t 3 /nobreak > nul
del response1.json response2.json response3.json 2>nul

endlocal
