import os
import random
from flask import Flask, request, jsonify
from flask_cors import CORS

# Importiere deine Noetiko-Sims (passe Pfad an, z. B. from .simulations.qutip_lindblad_model import lindblad_simulation)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Für Prod: Ersetze * durch Bubble-Domain

@app.route('/', methods=['GET'])
def health_check():
    return "NOETIKO CORE ONLINE", 200

@app.route('/calculate_entropy', methods=['POST'])
def calculate_entropy():
    try:
        data = request.get_json() or {}
        hrv = float(data.get('hrv', 50.0))
        amulett_active = data.get('amulett_active', False)
        duration = float(data.get('duration', 0.0))
        audio_used = data.get('audio_used', False)

        # Ersetze mit deiner realen Physik (z. B. lindblad_simulation(hrv))
        base_entropy = max(0, 100 - hrv)  # Platzhalter

        noise = random.uniform(-2.0, 2.0)
        current_entropy = base_entropy + noise

        boost_factor = 1.0
        if amulett_active:
            boost_factor += 0.10 + (min(duration, 5) * 0.02)
        if audio_used:
            boost_factor += 0.15

        coherence_score = (100 - current_entropy) * boost_factor
        projected_score = coherence_score * (1.20 if not amulett_active else 1.05)

        coherence_score = min(100, max(0, coherence_score))
        projected_score = min(100, max(0, projected_score))

        return jsonify({
            "entropy": round(100 - coherence_score, 1),
            "coherence_score": round(coherence_score, 1),
            "projected_score": round(projected_score, 1),
            "status": "calculated",
            "message": "Phase Lock established"
        }), 200

    except ValueError as ve:
        return jsonify({"error": f"Invalid input: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
