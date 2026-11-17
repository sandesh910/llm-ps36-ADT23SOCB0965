@echo off
cd /d "%~dp0"
call :main > push_log.txt 2>&1
type push_log.txt
pause
exit /b

:main
where git || (echo ERROR: Git not found & exit /b 1)
if not exist .git git init
git config user.name >nul || git config user.name "sandesh910"
git config user.email >nul || git config user.email "sandesh910@users.noreply.github.com"
git add .
git ls-files --cached | findstr /r /c:"^\.env$" && (echo ERROR: .env staged, aborting & git rm --cached .env & exit /b 1)
git commit -m "LLM mini project PS36: Research Paper Q&A"
git branch -M main
git remote remove origin 2>nul
git remote add origin https://github.com/sandesh910/llm-ps36-ADT23SOCB0965.git
git push -u origin main
echo PUSH_EXIT=%ERRORLEVEL%
exit /b
