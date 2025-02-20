# PDFRAG
# easy potentially multimodal PDFRAG

## Project implements semantic search in PDF files (drawings and technical documentation) using Flask, OpenAI GPT-4O and Qdrant. Supports uploading large PDF files (up to 1.5GB), stores upload history and enables semantic text search using embeddings.

### Features
Upload and process PDF files up to 1.5GB
Semantic search using GPT-4O
Web interface for file uploads and search
History of processed PDFs
Works with technical drawings and documentation

### App Structure:
pdf_search/                  # project root directory
├── .env                     # environment variables (api keys, settings)
├── requirements.txt         # python dependencies
├── static/                  # static files (css, js)
│   └── css/
│       └── style.css        # main stylesheet
├── templates/               # html templates
│   └── index.html           # main ui page
├── logs/                    # logs directory
│   └── app.log             # application logs
├── uploads/                 # uploaded pdf files
├── chunks/                  # temporary split pdf parts
└── app/                     # main application code
    ├── __init__.py         # flask app initialization
    ├── config.py           # app settings
    ├── models.py           # data models
    ├── pdf_processor.py    # pdf processing logic
    ├── routes.py           # api routes
    └── utils.py            # utility functions


### Setup
1. Create and activate virtual environment:
python3 -m venv venv source venv/bin/activate #Linux/Mac
venv\ Scripts\activate #Windows

2. Install dependencies:
pip install -r requirements. txt

3. Fill .env file settings:
OPENAI_ API_KEY=your_key
QDRANT HOST=localhost
QDRANT_PORT=6333

4. Use remote or run local Qdrant Database (for example using Docker):
docker run -p 6333:6333 qdrant/qdrant

5. Run Flask app:
flask run --port=5000
Web interface will be available at: http://localhost:5000    

#### Usage
1. Via web interface:
• Open http://localhost:5000
• Upload PDF file
• Select uploaded file from list
• Enter search query

2. Via API:
# Upload file
curl -X POST -F "file=@document.pdf" http: //localhost:5000/upload

# Search
curl-X POST http://localhost:5000/query \
-H "Content-Type: application/json" \
-d '{"query": "find shaft drawing", "collection":"collection_name"} '

#### System Requirements
• Python 3.8+
• 16GB RAM recommended
• Qdrant for vector storage
• OpenAl API key
