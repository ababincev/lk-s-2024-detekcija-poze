def train(model, num_epochs, train_loader, store_dict, test_loader=None):
    for epoch in range(num_epochs):
        train_running_loss = 0.0
        train_acc = 0.0

        model = model.train()
        
        for batch_num, (x, y) in enumerate(train_loader):
            
            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            loss = loss_function(y_hat, y)
            optimizer.zero_grad()
            # Do backpropagation algorithm and calculate all relevant gradients.
            loss.backward()
            # Update model parameters (weights and biases) w.r.t. computed gradients.
            optimizer.step()

            train_running_loss += loss.detach().item()
            train_acc += get_accuracy(y_hat=y_hat, y=y)

        
        epoch_loss = train_running_loss / batch_num
        epoch_acc = train_acc / batch_num
        
        store_dict = keep_store_dict(curve=epoch_loss, label='train_loss', store_dict=store_dict)
        store_dict = keep_store_dict(curve=epoch_acc, label='train_acc', store_dict=store_dict)
        print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
            %(epoch, epoch_loss, epoch_acc))

        if test_loader is not None:
            test_acc = test(model=model, test_loader=test_loader)
            store_dict = keep_store_dict(curve=test_acc, label='test_acc', store_dict=store_dict)
        
    return store_dict