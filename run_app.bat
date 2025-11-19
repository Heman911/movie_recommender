@echo off
cd /d C:\movie_recommender
call venv\Scripts\activate
streamlit run app\app.py
pause
