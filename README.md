# Geospatial Agent Demo

This repository demonstrates a Jupyter-based demo agent that:
- searches public imagery from the Microsoft Planetary Computer STAC,
- computes NDVI and other analytics,
- exposes geospatial operations as callable tools for an LLM (LangChain/OpenAI),
- includes visualizations and a small time-series example.

## How to run
1. Clone repo.
2. Create Python venv and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
