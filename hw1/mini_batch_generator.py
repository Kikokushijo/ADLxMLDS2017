def mini_batch_generator(batch_size=32):
    with open('fbank/train.ark', 'r+') as f:
        maps = mmap.mmap(f.fileno(), 0)
        prev_man, prev_wav = "", ""
        X_train, Y_train = [], []
        X_train_batch, Y_train_batch = [], []
        while True:
            index = 0
            row = maps.readline().decode('UTF-8').strip('\n')
            name, *feature = row.split(' ')
            try:
                man, wav, num = name.split('_')
            except:
                print('Error:', row)
            
            try:
                if man == prev_man and wav == prev_wav:
                    X_train.append([float(i) for i in feature])
                    Y_train.append(int(label_dict[name]))
            except:
                print('Error:', row)
            else:
                index += 1
                if X_train:
                    if len(X_train) >= 400:
                        X_train = np.array(X_train[:400])
                        Y_train = np.array(Y_train[:400])
                    else:
                        X_train = np.lib.pad(X_train, (0, 400-len(X_train)), 'edge')
                    Y_train = np_utils.to_categorical(Y_train, num_classes=48)
                    X_train_batch.append(X_train)
                    Y_train_batch.append(Y_train)
                    if index >= 32:
                        index = 0
                        yield(X_train, Y_train)
#                     yield (X_train, Y_train)
                prev_man, prev_wav = man, wav
                X_train = [[float(i) for i in feature]]
                Y_train = [int(label_dict[name])]