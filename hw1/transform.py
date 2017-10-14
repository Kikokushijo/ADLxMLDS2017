
reduce_dict = {}
with open('phones/48_39.map', 'r') as f:
	for line in f:
		original, convert = line.strip('\n').split('\t')
		reduce_dict[original] = convert

phone2char = {}
with open('48phone_char.map', 'r') as f:
    for line in f:
        phone, _, char = line.strip('\n').split('\t')
        phone2char[phone] = char

char2num = {}
num2char = {}
phone2num = {}
index = 0
for phone, char in phone2char.items():
    if phone != reduce_dict[phone]:
        phone2char[phone] = phone2char[reduce_dict[phone]]
    else:
        char2num[phone2char[phone]] = index
        num2char[index] = phone2char[phone]
        index += 1

for phone, char in phone2char.items():
    phone2num[phone] = char2num[phone2char[phone]]

with open('phone2num', 'w') as f:
    for phone, num in phone2num.items():
        f.write('%s %d\n' % (phone, num))

with open('num2char', 'w') as f:
    for num, char in num2char.items():
        f.write('%d %s\n' % (num, char))