# Import necessary modules
from flask import Flask, request, jsonify

# Create a Flask application
app = Flask(__name__)

# Sample data structure for storing patient records
patient_records = []

# Define a route to handle patient data submission
@app.route('/submit_patient_data', methods=['POST'])
def submit_patient_data():
    data = request.json  # Assuming the data is sent as JSON

    # Validate and process the patient data
    if 'patient_id' in data and 'eye_movement_data' in data:
        # Save the patient data to the records
        patient_records.append(data)
        return jsonify({"message": "Patient data submitted successfully"})
    else:
        return jsonify({"error": "Invalid data format"}), 400

# Define a route to retrieve patient records
@app.route('/get_patient_records', methods=['GET'])
def get_patient_records():
    return jsonify(patient_records)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)