@echo off
cd /d "C:\Users\kshed\OneDrive\Desktop\cluade code\betsightly-backend-devil"
"C:\Users\kshed\AppData\Local\Microsoft\WindowsApps\python.exe" services/league_adaptive_training.py >> logs/midnight_training.log 2>&1
