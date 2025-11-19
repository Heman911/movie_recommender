@echo off
rem open a new cmd window titled MovieRecommender and run commands there
start "MovieRecommender" cmd /k "cd /d C:\movie_recommender && call venv\Scripts\activate && echo Activated venv && streamlit run app\app.py"
