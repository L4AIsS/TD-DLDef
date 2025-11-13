

ERR_LAYERS = set()

file_path = '../../data/wei_fen.txt'
with open(file_path, 'r') as f:
    for line in f.readlines():
        data = line[3:-1]
        ERR_LAYERS.add(data)


print(f'The sum of layer {len(ERR_LAYERS)}')
for layer in ERR_LAYERS:
    print(layer)
