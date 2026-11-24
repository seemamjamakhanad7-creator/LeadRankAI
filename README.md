# LeadRank – P56 Lead Scoring Agent

A local Streamlit application with a Python backend, SQLite storage, a synthetic-data machine-learning demo, lead assignment, search demonstrations and optional Gemini explanations.

## 1. Requirements
- Windows 10/11
- Python 3.10 or newer
- Visual Studio Code
- Internet connection for installing packages and using Gemini

## 2. Open the project
1. Extract `LeadRank_P56.zip`.
2. Open VS Code.
3. Select **File → Open Folder** and choose `LeadRank_P56`.
4. Open the VS Code terminal (**Terminal → New Terminal**).

## 3. Install and run (Windows PowerShell)
```powershell
py -3 --version
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
If PowerShell blocks activation, use Command Prompt terminal and run:
```bat
.venv\Scripts\activate.bat
pip install -r requirements.txt
streamlit run app.py
```
Or double-click `run_windows.bat`.

## 4. Configure Gemini
1. Copy `.env.example` and rename the copy to `.env`.
2. Open `.env` and replace `your_gemini_api_key_here` with your key.
3. Keep `.env` private. Do not upload it to GitHub or include it in screenshots.
4. Restart Streamlit after changing `.env`.

The app still works without Gemini: it falls back to a local rule-based recommendation. Gemini is used only to explain a selected lead and suggest an action; it does not contact customers.

## 5. Features
- Dark dashboard and summary metrics
- Add and score leads
- Scikit-learn Random Forest probability estimate
- SQLite persistence
- Lead search, filtering and CSV export
- Greedy load-balancing assignment demonstration
- BFS, DFS, Uniform Cost, Greedy and A* toy search demonstrations
- Optional Gemini explanation
- Human approval and activity log
- Responsible AI and model-evaluation page

## 6. Important project limitations (be honest in your viva)
- Training records are synthetic demonstration data, not actual customer conversion history.
- The displayed AUC and accuracy are measured on a held-out split of that synthetic data. They are not a real-world guarantee and must not be described as validation on real customers.
- The search lab is a small educational feature-selection example.
- Assignment is a greedy load-balancing baseline, not a full OR-Tools CSP implementation.
- The app does not automatically send emails, make calls, or take irreversible actions.
- The handbook lists LangGraph, AutoGen, CrewAI, Prolog, OR-Tools, RAG, Q-learning, MARL and other extensions. They are not all implemented in this compact deadline-focused build. Do not claim they are implemented unless you add and test them.

## 7. Suggested demonstration
1. Open Overview.
2. Add a sample lead (e.g. a technology company, medium size, budget ₹80,000, 12 visits, 5 email opens, demo requested).
3. Show its score and recommendation.
4. Open Lead Explorer and show the saved database record.
5. Run Assignment with the available representatives.
6. Open Algorithm Lab and compare the search strategies.
7. Open AI Assistant and generate a Gemini explanation.
8. Show Ethics & Evaluation and explain the synthetic-data limitation and human review.

## 8. Suggested report description
LeadRank is a sales decision-support prototype that estimates lead-conversion likelihood from business-relevant engagement and company information. It assigns a score from 0 to 100, groups leads into High, Medium and Low priority, stores records in SQLite, and presents advisory next steps. A human reviewer remains responsible for outreach decisions.

## 9. Troubleshooting
- `python is not recognized`: install Python 3.10+ and select “Add Python to PATH”, then reopen VS Code.
- `streamlit is not recognized`: activate `.venv` and run `python -m pip install -r requirements.txt`.
- Gemini not connected: check `.env` filename, API key, internet connection and restart Streamlit.
- Port busy: run `streamlit run app.py --server.port 8502`.
