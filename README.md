# GitHub Tutor (GPT-5) — Hackathon Space

## How to run locally
```bash
pip install -r requirements.txt
$env:AIML_API_KEY="8b7b1b6b0f3143448bb1374e243e7091"
streamlit run app.py
```

## Deploy to Hugging Face Spaces
1. Create account → New Space → **SDK: Streamlit**.
2. Upload `app.py`, `requirements.txt`, `questions.json`.
3. In Space → **Settings → Variables and secrets**:
   - Add `OPENAI_API_KEY` with your GPT-5 key.
4. Click **Restart** if needed; open the app URL.

## Notes
- The file system is ephemeral. Summaries are kept in session only (demo-friendly).
- Edit `questions.json` to add/remove problems for the demo.
