# Fashion-MNIST ile Kıyafet Sınıflandırması

[Fashion-MNIST](https://github.com/cansuyildiim/fashion-mnist), kıyafet sınıflandırması için bir veri kümesidir. 10 farklı sınıftan oluşan 7.000 resim içerir ve bunlar eğitimde 60.000 ve testte 10.000 resim olarak ayrılmıştır. Derin öğrenme modelleri, [resmi ölçüm](https://github.com/cansuyildiriim/fashion-mnist#benchmark) olan yaklaşık %95 doğruluk oranına ulaşmaktadır. Benzer bir doğruluk elde eden ve "verimli" olan modelleri eğitmek ve bunları düzenli bir bilgisayarda kullanmak istiyoruz.

Bu keşif çalışmasında üç farklı hedefimiz var:

1. Farklı model mimarilerini ve eğitim stratejilerini kullanarak farklı modeller eğitmek.
2. Farklı deneylerin sonuçlarını tartışmak.
3. Eğitilmiş modelleri bir web kamerası ile bir demo kullanarak test etmek.

## Kurulum

En az **Python 3.6** sürümüne sahip izole bir Python ortamı kullanmanızı, örneğin [venv](https://docs.python.org/3/library/venv.html) veya [conda](https://docs.conda.io/en/latest/) öneriyoruz. Ardından, aşağıdaki kodları kullanarak kurulumu yapabilirsiniz:

```bash
git clone https://github.com/cansuyildiriim/Fashion-MNIST-PyTorch.git
cd Fashion-MNIST-PyTorch
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```

### PyTorch Kurulumu

PyTorch'un kurulumu her platform için farklı olduğundan, lütfen [PyTorch kurulum kılavuzuna](https://pytorch.org/get-started/locally/) göz atın.

## Kullanım

Kurulum işlemi tamamlandıktan sonra, farklı deneylerin sonuçlarını çoğaltmak için `train_fashionMNIST.py` betiğini kullanın.


### Proje Yapısı

Proje beş farklı klasör içerir:

- **data** : Bu dizin `train_fashionMNIST.py` betiği ilk kez çalıştırıldığında oluşturulur. Fashion MNIST'in eğitim ve test veri kümelerini içerir.
- **demo** : Bu dizin demo için tüm kodları içerir.
- **experiments** : Bu dizin `train_fashionMNIST.py` betiği ilk kez çalıştırıldığında oluşturulur. Deneylerin sonuçlarını içerir. Eğitilmiş modelleri bu dizine koymak onları demo ile kullanmanızı sağlar.
- **images** : Bu dizin bu README dosyasında kullanılan resimleri içerir.
- **models** : Bu dizin modellerin mimarisi ve etiketlerin tanımını içerir.

## Demo

Bu projede gerçek zamanlı görüntü işleme için bir demo uygulaması bulunmaktadır. Demo uygulamasını çalıştırmak için run_inference.py adlı bir Python betiği kullanılıyor. Bu betik eğitilmiş modelleri kullanarak görüntü sınıflandırması yapıyor ve sonuçları ekranda gösteriyor.

Demo'yu çalıştırmak için aşağıdaki komutları kullanın:

```bash
cd demo
python3 -m venv .env
source .env/bin/activate
pip install -r requirements
```

```bash
cd demo
python run_inference.py --model SimpleCNNModel --weights_path ../demo/experiments/01_SimpleCNNModel/model_best.pth.tar --display_input --display_fps --device 0
```