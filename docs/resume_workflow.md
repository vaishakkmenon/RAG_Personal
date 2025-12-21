# Resume Workflow: The Master Profile Strategy

To ensure your RAG system always gives the most comprehensive answers about your experience, we have adopted a **Superset Strategy**.

## The Concept

Instead of ingesting specific, formatted resumes (e.g., "Resume 2025.md"), which are often constrained by page limits and formatting, we use a single **Master Profile**.

*   **Master Profile (`data/mds/master_profile.md`)**: A "living document" that contains *everything* you have done. It is written in dense, narrative prose to maximize retrieval quality. It has no page limit.
*   **Targeted Resumes**: These are standard PDF/Word documents you send to recruiters. They are *subsets* of the Master Profile. **We do NOT ingest these into RAG** because they are redundant and less detailed.

## How to Update Your Resume (Future)

When you start a new job, complete a new project, or earn a new cert:

1.  **Do NOT** just drag-and-drop your new 1-page resume file into `data/mds/`.
2.  **Instead**, open `data/mds/master_profile.md`.
3.  **Append** the new information to the relevant section (Experience, Projects, etc.).
    *   *Tip*: Write it as a story ("I built X using Y..."), not just bullet points. This helps the AI understand context better.
4.  **Save** the file.
5.  **Re-ingest** (if your server isn't auto-watching):
    ```bash
    curl -X POST http://localhost:8000/ingest -H "X-API-Key: your-key"
    ```

## Handling "Old" Data

If you decide to remove an experience from your public resume (e.g., it's 10 years old), **keep it in the Master Profile**.
*   **Why?**: You want your RAG system to remember it ("Do I have experience with Perl?"), even if you don't advertise it on your current 1-pager.
*   **Tagging**: You can move it to a "Historical Experience" section within the same file if you want to organize it, but keeping it in the file ensures it remains searchable.

## Architecture

*   `data/mds/master_profile.md` -> **Source of Truth** (Ingested)
*   `data/mds/resume--*.md.old` -> **Archives** (Ignored by Ingestion)
