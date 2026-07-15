# 🦜 LangChain: Summarize Text From YT or Website

A Flask + HTML/CSS/JS app that summarizes website articles or YouTube videos
using LangChain's map-reduce summarization chain, with optional translation
into English, French, Persian, Chinese, or Arabic.

## Local development

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:5000

## Deployment (Vercel)

1. Push this repo to GitHub.
2. In Vercel: **Add New → Project → Import** this repo.
3. Vercel auto-detects Flask from `requirements.txt` — no extra config needed.
4. Static assets are served from `public/` (not Flask's `static/` folder,
   per Vercel's Flask deployment requirements).
