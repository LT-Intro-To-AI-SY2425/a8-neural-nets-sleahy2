import random
import sys
try:
    from neural import NeuralNet  # assumes your class is in neural.py
except ImportError:
    raise ImportError("The 'neural.py' file or 'NeuralNet' class is missing. Ensure it is implemented and in the same directory.")

def parse_line(line):
    parts = line.strip().split(",")
    if len(parts) != 5:
        return None
    try:
        inputs = list(map(float, parts[:4]))
    except ValueError:
        return None
    
    label = parts[4]
    if label == "Iris-setosa":
        output = [1, 0, 0]
    elif label == "Iris-versicolor":
        output = [0, 1, 0]
    elif label == "Iris-virginica":
        output = [0, 0, 1]
    else:
        return None
    
    return inputs, output

def normalize(data):
    num_features = len(data[0][0])
    mins = [float("inf")] * num_features
    maxs = [float("-inf")] * num_features

    for inputs, _ in data:
        for i in range(num_features):
            mins[i] = min(mins[i], inputs[i])
            maxs[i] = max(maxs[i], inputs[i])

    normalized_data = []
    for inputs, output in data:
        normalized_inputs = [
            (inputs[i] - mins[i]) / (maxs[i] - mins[i]) if maxs[i] != mins[i] else inputs[i]
            for i in range(num_features)
        ]
        normalized_data.append((normalized_inputs, output))
    
    return normalized_data

data = []
try:
    with open("iris.data", "r") as f:  # Ensure this path is correct for your dataset
        for line in f:
            if line.strip():
                parsed = parse_line(line)
                if parsed is not None:
                    data.append(parsed)
    if not data:
        raise ValueError("Dataset is empty or improperly formatted.")
    print(f"Loaded {len(data)} valid examples.")
except FileNotFoundError:
    print("The dataset file 'iris.data' was not found. Please ensure the file exists and the path is correct.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while loading the dataset: {e}")
    sys.exit(1)

data = normalize(data)
random.shuffle(data)

# Train-test split
split_index = int(0.8 * len(data))
train_data = data[:split_index]
test_data = data[split_index:]

# Create and train the network
try:
    nn = NeuralNet(n_input=4, n_hidden=5, n_output=3)
    nn.train(train_data, learning_rate=0.5, momentum_factor=0.1, iters=1000, print_interval=100)
except Exception as e:
    print(f"Training failed: {e}")
    sys.exit(1)

# Test
try:
    correct = 0
    for inputs, expected in test_data:
        output = nn.evaluate(inputs)
        predicted_index = output.index(max(output))
        actual_index = expected.index(max(expected))
        if predicted_index == actual_index:
            correct += 1

    accuracy = correct / len(test_data)
    print(f"Accuracy on test set: {accuracy:.2f}")
except Exception as e:
    print(f"Testing failed: {e}")