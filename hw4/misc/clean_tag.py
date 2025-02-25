import os
import json
hair_dict = {}
eyes_dict = {}
common_dict = {}
hair_candidate = ['orange hair', 'white hair', 'aqua hair', 'gray hair',
                  'green hair', 'red hair', 'purple hair', 'pink hair',
                  'blue hair', 'black hair', 'brown hair', 'blonde hair']
eyes_candidate = ['gray eyes', 'black eyes', 'orange eyes',
                  'pink eyes', 'yellow eyes', 'aqua eyes', 'purple eyes',
                  'green eyes', 'brown eyes', 'red eyes', 'blue eyes']
with open('../hw4_dataset/tags_clean.csv') as f:
    cont = f.readlines()
    cont = [x.strip() for x in cont]
    for line in cont:
        no, tags = line.split(',')
        tags = tags.split('\t')

        for tag in tags:
            if 'hair' in tag:
                cand, _ = tag.split(':')
                if cand in hair_candidate:
                    hair_dict[no] = cand
                    print(no, cand)
            if 'eyes' in tag:
                cand, _ = tag.split(':')
                if cand in eyes_candidate:
                    eyes_dict[no] = cand
                    print(no, cand)

# print(len(hair_dict), len(eyes_dict))

for key, value in hair_dict.items():
    if key in eyes_dict:
        common_dict[key] = [value, eyes_dict[key]]

with open('dict/hair.json', 'w+') as f:
    json.dump(hair_dict, f)

with open('dict/eyes.json', 'w+') as f:
    json.dump(eyes_dict, f)

with open('dict/common.json', 'w+') as f:
    json.dump(common_dict, f)

hair2id = {}
id2hair = {}
for index, color in enumerate(hair_candidate):
    hair2id[color] = index
    id2hair[index] = color

with open('dict/hair2id.json', 'w+') as f:
    json.dump(hair2id, f)

with open('dict/id2hair.json', 'w+') as f:
    json.dump(id2hair, f)

eyes2id = {}
id2eyes = {}
for index, color in enumerate(eyes_candidate):
    eyes2id[color] = index
    id2eyes[index] = color

with open('dict/eyes2id.json', 'w+') as f:
    json.dump(eyes2id, f)

with open('dict/id2eyes.json', 'w+') as f:
    json.dump(id2eyes, f)

# print(len(common_dict))