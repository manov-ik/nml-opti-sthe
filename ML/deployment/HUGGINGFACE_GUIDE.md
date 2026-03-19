# How to Deploy HEXP.io

This guide covers deploying the AI backend to Hugging Face and the frontend to Netlify.

## 1. Deploy the AI Backend (Hugging Face Spaces)
1.  **Create a Space**: Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2.  **Name**: `heat-exchanger-optimizer` (or similar).
3.  **SDK**: Select **Gradio**.
4.  **Upload Files**: Go to "Files and Versions" and upload:
    - `app.py`
    - `requirements.txt`
    - `random_forest_heat_exchanger_model.pkl`
    - `preprocessor_dual_fluid.pkl`

## 2. Deploy the Frontend (Netlify)
1.  **Preparation**: Your `index.html` (in `frontend/`) now uses a placeholder `[[HF_SPACE_URL]]`.
2.  **Netlify Settings**:
    - **Go to**: Site Settings > Build & Deploy > Environment > Environment variables.
    - **Add variable**: `HF_SPACE_URL` = `your-username/your-space-name`.
3.  **Build Command**:
    - **Go to**: Site Settings > Build & Deploy > Continuous Deployment > Build settings.
    - **Build command**: `sed -i "s|\[\[HF_SPACE_URL\]\]|${HF_SPACE_URL}|g" index.html`
    - **Publish directory**: `.` (or wherever your `index.html` is).

> [!TIP]
> This command automatically replaces the placeholder in your code with the actual URL from your Netlify settings during every deployment.

## 3. Using the AI Optimization
- **Predict Mode**: Enter all values.
- **Optimize Mode**: Enter **-1** for any parameter (like Shell Diameter or No. of Tubes) you want the AI to find for you.
- **Target Q**: Set your desired heat transfer rate. The AI will simulate 1000+ designs and show you the top 3 that meet your target.

> [!NOTE]
> All animations have been removed from the frontend for a faster, cleaner professional experience as requested.
