from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)

# Model structure
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Load trained model
model = Net()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Mapping numeric labels to species names
species = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    x = torch.tensor([data], dtype=torch.float32)
    with torch.no_grad():
        output = model(x)
        _, pred = torch.max(output, 1)
    # Convert numeric label to species name
    predicted_species = species[int(pred.item())]
    return jsonify({"prediction": predicted_species})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
