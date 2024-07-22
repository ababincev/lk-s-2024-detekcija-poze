def gaussian_initializer(mean=0.0, stddev=0.01):
    def _initializer(tensor):
        return nn.init.normal_(tensor, mean=mean, std=stddev)
    return _initializer

def constant_initializer(value=0.0):
    def _initializer(tensor):
        return nn.init.constant_(tensor, value)
    return _initializer