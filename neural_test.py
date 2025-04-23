from neural import NeuralNet

print("\n--- PART 1: XOR with 2 Hidden Nodes ---")
xor_training_data = [([1, 1], [0]), ([1, 0], [1]), ([0, 1], [1]), ([0, 0], [0])]
xorn_2hidden = NeuralNet(2, 2, 1)
xorn_2hidden.train(xor_training_data)
print(xorn_2hidden.test_with_expected(xor_training_data))

print("\n--- PART 2: Repeat Training to See Convergence Differences ---")
for i in range(3):
    print(f"\nTraining run #{i + 1}")
    temp_xor = NeuralNet(2, 2, 1)
    temp_xor.train(xor_training_data)
    print(temp_xor.test_with_expected(xor_training_data))

print("\n--- PART 3: XOR with 8 Hidden Nodes ---")
xorn_8hidden = NeuralNet(2, 8, 1)
xorn_8hidden.train(xor_training_data)
print(xorn_8hidden.test_with_expected(xor_training_data))

print("\n--- PART 4: XOR with 1 Hidden Node (Should Fail to Learn XOR) ---")
xorn_1hidden = NeuralNet(2, 1, 1)
xorn_1hidden.train(xor_training_data)
print(xorn_1hidden.test_with_expected(xor_training_data))

print("\n--- PART 5: Voter Data ---")
rn_training_data = [
    ([1, 0, 1, 0, 0, 0], [1]),
    ([1, 0, 1, 1, 0, 0], [1]),
    ([1, 0, 1, 0, 1, 0], [1]),
    ([1, 1, 0, 0, 1, 1], [1]),
    ([1, 1, 1, 1, 0, 0], [1]),
    ([1, 0, 0, 0, 1, 1], [1]),
    ([1, 0, 0, 0, 1, 0], [0]),
    ([0, 1, 1, 1, 0, 1], [1]),
    ([0, 1, 1, 0, 1, 1], [0]),
    ([0, 0, 0, 1, 1, 0], [0]),
    ([0, 1, 0, 1, 0, 1], [0]),
    ([0, 0, 0, 1, 0, 1], [0]),
    ([0, 1, 1, 0, 1, 1], [0]),
    ([0, 1, 1, 1, 0, 0], [0]),
]
nn = NeuralNet(6, 1, 1)
nn.train(rn_training_data)
print(nn.evaluate([1, 1, 1, 1, 1, 1]))
print(nn.evaluate([0, 0, 0, 0, 0, 0]))
print(nn.evaluate([1, 0, 0, 0, 0, 0]))
print(nn.evaluate([0, 1, 0, 0, 0, 0]))
print(nn.evaluate([0, 0, 1, 0, 0, 0]))
print(nn.evaluate([0, 0, 0, 1, 0, 0]))
print(nn.evaluate([0, 0, 0, 0, 1, 0]))
print(nn.evaluate([0, 0, 0, 0, 0, 1]))
print(nn.evaluate([0, 1, 1, 1, 1, 1]))
print(nn.evaluate([1, 0, 1, 1, 1, 1]))
print(nn.evaluate([1, 1, 0, 1, 1, 1]))
print(nn.evaluate([1, 1, 1, 0, 1, 1]))
print(nn.evaluate([1, 1, 1, 1, 0, 1]))
print(nn.evaluate([1, 1, 1, 1, 1, 0]))
print(nn.evaluate([0, 1, 1, 1, 0, 1]))
print(nn.evaluate([1, 0, 0, 0, 1, 0]))