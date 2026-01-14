from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import multiprocessing
from config import Config
from indexer import process_video
from retriever import query_videorag

app = Flask(__name__)
CORS(app)

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    save_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    file.save(save_path)
    
    # Start Indexing in Background
    p = multiprocessing.Process(target=process_video, args=(save_path,))
    p.start()
    
    return jsonify({"message": "Upload successful, indexing started.", "filename": filename})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query')
    result = query_videorag(query)
    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000, debug=True)