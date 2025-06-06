@echo off
echo 🔧 Setting up virtual environment...
python -m venv venv

echo 📦 Installing dependencies...
call venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

echo ✅ Setup complete.
echo Run the app with:
echo venv\Scripts\activate && python main.py
pause
