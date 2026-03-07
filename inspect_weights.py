import json
with open('public/models/isl_model/weights.json', 'r') as f:
    weights = json.load(f)
for i, layer in enumerate(weights):
    w = layer['weights']
    b = layer['biases']
    print(f"Layer {i}: {layer['name']}, weight shape: {len(w)}x{len(w[0]) if isinstance(w[0], list) else 1}, bias shape: {len(b)}")
