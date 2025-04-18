import torch.nn as nn


class SimpleCNNModel(nn.Module):
    """ SimpleCNNModel, bir temel olarak kullanılabilecek basit bir CNN modelidir.
    
        Model Yapısı:
            2x Evrişim Katmanı:
                - ReLU Aktivasyonu
                - Batch Normalizasyon
                - Uniform Xavier Ağırlıkları
                - Max Pooling
      
            1x Tam Bağlı Katman:
                - ReLU aktivasyonu
      
            1x Tam Bağlı Katman:
                - Çıkış Katmanı
    """

    def __init__(self):
        super(SimpleCNNModel, self).__init__()

        self.cnn1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2
        )
        self.relu1 = nn.ReLU()
        self.norm1 = nn.BatchNorm2d(32)
        nn.init.xavier_uniform_(self.cnn1.weight)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=2
        )
        self.relu2 = nn.ReLU()
        self.norm2 = nn.BatchNorm2d(64)
        nn.init.xavier_uniform_(self.cnn2.weight)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(4096, 4096)
        self.fcrelu = nn.ReLU()

        self.fc2 = nn.Linear(4096, 10)

    def forward(self, x):
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.norm1(out)

        out = self.maxpool1(out)

        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.norm2(out)

        out = self.maxpool2(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.fcrelu(out)

        out = self.fc2(out)
        return out
