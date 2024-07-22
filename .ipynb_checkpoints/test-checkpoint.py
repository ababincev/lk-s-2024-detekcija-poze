def test(model, test_loader):
    # Put the model in evaluation mode. 
    # Tells the model not to compute gradients. Increases inference speed.
    model.eval()
    test_acc = 0.0
    with torch.no_grad():
        for batch_num, (x, y) in enumerate(test_loader, 0):
            # Put the data to the appropriate device.
            x = x.to(device)
            y = y.to(device)
            # Do inference. Forwad pass with the model.
            y_hat = model(x)
            test_acc += get_accuracy(y_hat, y)
    return test_acc / batch_numdef test(model, test_loader):
    # Put the model in evaluation mode. 
    # Tells the model not to compute gradients. Increases inference speed.
    model.eval()
    test_acc = 0.0
    with torch.no_grad():
        for batch_num, (x, y) in enumerate(test_loader, 0):
            # Put the data to the appropriate device.
            x = x.to(device)
            y = y.to(device)
            # Do inference. Forwad pass with the model.
            y_hat = model(x)
            test_acc += get_accuracy(y_hat, y)
    return test_acc / batch_num