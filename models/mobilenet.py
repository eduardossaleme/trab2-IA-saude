from torch import nn


class MyMobilenet (nn.Module):

    def __init__(self, mobilenet, num_class, neurons_reducer_block=256, freeze_conv=False, p_dropout=0.5, n_feat_conv=1280):

        super(MyMobilenet, self).__init__()


        self.features = nn.Sequential(*list(mobilenet.children())[:-1])

        # freezing the convolution layers
        if freeze_conv:
            for param in self.features.parameters():
                param.requires_grad = False

        # Feature reducer
        if neurons_reducer_block > 0:
            self.reducer_block = nn.Sequential(
                nn.Linear(n_feat_conv, neurons_reducer_block),
                nn.BatchNorm1d(neurons_reducer_block),
                nn.ReLU(),
                nn.Dropout(p=p_dropout)
            )
        else:
            self.reducer_block = None

        if neurons_reducer_block > 0:
            self.classifier = nn.Linear(neurons_reducer_block, num_class)
        else:
            self.classifier = nn.Linear(n_feat_conv, num_class)

    def forward(self, img):

        x = self.features(img)
        x = x.mean([2, 3])

        x = x.view(x.size(0), -1) # flatting
        if self.reducer_block is not None:
            x = self.reducer_block(x)  # feat reducer block


        return self.classifier(x)
