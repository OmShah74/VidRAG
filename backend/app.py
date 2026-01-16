from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import multiprocessing
from config import Config
from indexer import process_video
from retriever import query_videorag

app = Flask(__name__)

# 1. Enable CORS for the specific Frontend URL to avoid blocking
CORS(app, resources={r"/*": {"origins": "*"}})

# 2. Increase Upload Limit to 16GB (Critical for Videos)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 * 1024 

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    filename = secure_filename(file.filename)
    save_path = os.path.join(Config.UPLOAD_FOLDER, filename)
    
    try:
        file.save(save_path)
    except Exception as e:
        return jsonify({"error": f"Failed to save file: {str(e)}"}), 500
    
    # Start Indexing in Background
    p = multiprocessing.Process(target=process_video, args=(save_path,))
    p.start()
    
    return jsonify({"message": "Upload successful, indexing started.", "filename": filename})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
        
    query = data.get('query')
    try:
        result = query_videorag(query)
        return jsonify(result)
    except Exception as e:
        print(f"Error processing chat: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run on 0.0.0.0 to ensure it binds to all interfaces (IPv4/IPv6)
    app.run(host='0.0.0.0', port=5000, debug=True)