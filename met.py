import json
import logging
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        with open('model/evaluation_metrics.json', 'r') as f:
            metrics = json.load(f)
        return jsonify(metrics)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return jsonify({"error": "File not found"}), 404
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON: {e}")
        return jsonify({"error": "Error parsing JSON"}), 400
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return jsonify({"error": "Unexpected error"}), 500

if __name__ == '__main__':
    app.run(port=4300)