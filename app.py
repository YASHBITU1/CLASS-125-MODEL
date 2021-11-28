from flask import Flask,jsonify,request
from prediction import getPrediction
app = Flask(__name__)
@app.route("/predictdigit",methods = ["POST"])
def predictDigit():
    image = request.files.get("digit")
    result = getPrediction(image)
    return jsonify({
        "prediction":result
    }),200
if(__name__ == "__main__"):
    app.run(debug=True)

