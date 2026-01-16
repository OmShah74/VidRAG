# VidRAG: Video Retrieval-Augmented Generation

VidRAG is a professional-grade multi-modal framework designed for semantic video analysis and conversational interaction. By integrating advanced computer vision, natural language processing, and graph-based knowledge extraction, VidRAG enables users to query video content with unprecedented precision.

## Technical Overview

The platform employs a multi-channel indexing and retrieval architecture, ensuring that queries are resolved across visual, textual, and relational dimensions.

### Core Processing Pipeline

1.  **Temporal Segmentation**: Videos are processed in 30-second segments to maintain granular temporal context.
2.  **Multi-Modal Feature Extraction**:
    *   **Visual Channel**: Utilizes **OpenCLIP (ViT-B-32)** to generate dense vector embeddings for keyframes.
    *   **Textual Channel**: Integrates **OpenAI Whisper** for high-accuracy speech-to-text transcription.
    *   **Semantics**: Employs **MiniCPM-V 2.6** (Vision-Language Model) for generating detailed scene descriptions and visual captions.
3.  **Graph Construction**: Dynamically extracts entities and relationships using LLM-guided parsing, building a semantic knowledge graph with **NetworkX**.
4.  **Vector Persistence**: Stores high-dimensional embeddings in a specialized **nano-vectordb**, supporting fast top-k similarity searches.

### Retrieval Mechanism

The retrieval engine uses a "Query Reformulation" strategy:
*   **Visual Retrieval**: Searches for visual matches using CLIP-based text-to-image embeddings.
*   **Keyword Retrieval**: Executes dense vector search on transcriptions and visual captions using OpenAI's `text-embedding-3-small`.
*   **Context Fusion**: Results from all channels are de-duplicated, chronologically sorted, and injected into the LLM prompt context for grounded response generation.

## Project Structure

```text
vidrag/
├── backend/                # Flask-based high-performance API
│   ├── services/           # Implementation of AI, Graph, and Vector engines
│   │   ├── ai_engine.py    # CLIP, VLM, and Whisper orchestration
│   │   ├── graph_engine.py # Knowledge graph management
│   │   ├── llm_engine.py   # OpenAI GPT-4o-mini integration
│   │   └── vector_engine.py# Similarity search management
│   ├── data/               # Persistent storage for videos and indexes
│   ├── indexer.py          # Background indexing logic
│   ├── retriever.py        # Query processing and retrieval pipeline
│   └── app.py              # Main Flask server entry point
├── frontend/               # Next.js 15+ Web Application
│   ├── src/app/            # Modular React components and pages
│   └── globals.css         # Modern styling with Tailwind CSS v4
└── README.md               # Project documentation
```

## Setup and Deployment

### Hardware Requirements
*   **GPU Interaction**: Highly recommended for VLM-based scene analysis (MiniCPM-V).
*   **CPU Fallback**: Automatically detected; visual analysis will switch to basic mode to conserve resources.

### Installation Instructions

#### Backend Initialisation
1.  Navigate to the backend directory: `cd backend`
2.  Install dependencies: `pip install -r requirements.txt`
3.  Ensure your `OPENAI_API_KEY` is configured in `config.py`.
4.  Launch the server: `python app.py`

#### Frontend Initialisation
1.  Navigate to the frontend directory: `cd frontend`
2.  Install packages: `npm install`
3.  Start the development server: `npm run dev`

## API Documentation

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/upload` | `POST` | Uploads video and triggers asynchronous background indexing. |
| `/api/chat` | `POST` | Processes natural language queries with retrieval augmentation. |

---

## License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
