import json

with open('../dict/hair2id.json') as f:
    hair2id = json.load(f)
        
with open('../dict/eyes2id.json') as f:
    eyes2id = json.load(f)

with open('../testing.txt', 'w+') as f:
    count = 1
    for key in hair2id:
        f.write('%d,%s\n' %(count, key))
        count += 1

    for key in eyes2id:
        f.write('%d,%s\n' %(count, key))
        count += 1

    for key in hair2id:
        for key2 in eyes2id:
            f.write('%d,%s %s\n' %(count, key, key2))
            count += 1