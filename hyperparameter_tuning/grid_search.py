from fcnn import FCNN

fcnet = FCNN()
scores = {}
n_hidden = [60, 70, 80]
dropouts = [0.5, 0.75, 1.0]
epochs = [15, 30, 45]
n_layers = [1, 2, 3]

for epoch in epochs:
    for hidden in n_hidden:
        for dropout in dropouts:
            for layers in n_layers:
                params = 'epoch:{},hidden:{},dropout:{}'.format(epoch, hidden, dropout)
                uid, train_acc, test_acc, val_acc = fcnet.train(n_epochs=epoch, dropout_prob=dropout, n_hidden=hidden,
                                                                n_layers=layers)
                print('Job: {} completed'.format(str(uid)))
                scores[params] = [train_acc, test_acc, val_acc]

print(scores)
