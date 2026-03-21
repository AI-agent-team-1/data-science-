# Запуск бота через Python из venv проекта (чтобы подхватывались все зависимости)
$ProjectRoot = Split-Path -Parent $PSScriptRoot
& "$ProjectRoot\venv\Scripts\python.exe" "$ProjectRoot\bot.py"
