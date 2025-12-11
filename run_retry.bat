@echo off
echo Starting script at %date% %time% > C:\Temp\batch_debug.txt
cd /d C:\Users\vaish\Personal\Other\Coding\RAG_Personal
echo Working directory: %CD% >> C:\Temp\batch_debug.txt
echo Running Python... >> C:\Temp\batch_debug.txt
C:\Python313\python.exe instance-create.py >> C:\Temp\batch_debug.txt 2>&1
echo Script ended at %date% %time% >> C:\Temp\batch_debug.txt
