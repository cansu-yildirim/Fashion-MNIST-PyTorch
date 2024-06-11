import argparse
import datetime
import json
import os
import random
import shutil
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms, datasets, models

import numpy as np
from prefetch_generator import BackgroundGenerator

from models.SimpleCNNModel import SimpleCNNModel
import utils


    # Python'da komut satırı argümanlarını analiz etmek ve işlemek için kullanılan argparse modülünü kullanarak bir komut satırı arayüzü oluşturmayı hedefliyoruz

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

    # Model Seçimi

parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=["SimpleCNNModel", "ResNet18"],
    help="Eğitilecek modelin adı.",
)

    # Deney Adı

parser.add_argument(
    "--experiment_name",
    type=str,
    default="",
    help="Deneyin adı, deneyin izlenmesi ve yeniden başlatılması için tüm verilerin saklanacağı klasöre isim vermek için kullanılacaktır.",
)

    # Doğrulama Seti Yüzdesi

parser.add_argument(
    "--percentage_validation_set",
    type=int,
    default=10,
    choices=range(0, 101),
    metavar="[0-100]",
    help="Eğitim setinden doğrulama seti olarak kullanılacak veri yüzdesi.",
)

    # Erken Durdurma (Patience)

parser.add_argument(
    "--patience",
    type=int,
    default=-1,
    help="Doğrulama setinin doğruluğunda iyileşme olmadan kaç epoch devam edileceği. Erken durdurmayı devre dışı bırakmak için -1 kullanın.",
)

    # Batch Size, her bir eğitim adımında kaç veri örneğinin kullanılacağınu belirtir

parser.add_argument(
    "--batch_size",
    type=int,
    default=256,
    choices=range(1, 1025),
    metavar="[1-1024]",
    help="Batch boyutu.",
)

    # Epoch Sayısı, maksimum eğitim döngüsü sayısını belirtir

parser.add_argument(
    "--num_epochs",
    type=int,
    default=30,
    choices=range(1, 1001),
    metavar="[1-1000]",
    help="Maksimum epoch sayısı.",
)

    # Çalışan Sayısı, verileri yüklemek için kullanılacak işçi sayısı

parser.add_argument(
    "--num_workers",
    type=int,
    default=1,
    choices=range(1, 65),
    metavar="[1-64]",
    help="Çalışan sayısı.",
)

    # Rasgele Tohum, --manual_seed argümanı, rastgelelik için kullanılacak tohum değerini belirtir

parser.add_argument("--manual_seed", type=int, default=42, help="Rastgele tohum.")

    # Öğrenme Oranı, --learning_rate argümanı, başlangıç öğrenme oranını belirtir

parser.add_argument(
    "--learning_rate", required=True, type=float, help="Başlangıç öğrenme oranı."
)

    # Momentum, --momentum argümanı, momentum değerini belirtir ve varsayılan değeri 0'dır
    # Ağırlık Azaltma, --weight_decay argümanı, ağırlık azaltma (regularizasyon) değerini belirtir. Varsayılan değeri 0'dır
    # --momentum, --weight_decay, --nesterov: Optimizasyon parametreleridir

parser.add_argument("--momentum", default=0, type=float, help="Momentum.")
parser.add_argument("--weight_decay", default=0, type=float, help="Ağırlık azalma.")
parser.add_argument("--nesterov", help="Nesterov momentumunu etkinleştir.", action="store_true")

    # Sınıf Yolu, --path_classes argümanı, Fashion MNIST sınıflarını içeren json dosyasının yolunu belirtir

parser.add_argument(
    "--path_classes",
    default=os.path.join("models", "classes.json"),
    type=str,
    help="Fashion MNIST sınıflarını içeren json dosyasının yolu.",
)
parser.add_argument(
    "--random_crop",
    help="Dönüşüm listesinin başında rastgele kırpma ekleyin.",
    action="store_true",
)
parser.add_argument(
    "--random_erasing",
    help="Dönüşüm listesinin sonunda rastgele silme ekleyin.",
    action="store_true",
)
parser.add_argument(
    "--convert_to_RGB",
    help="Görüntüyü 3 kanalda tekrarlayarak RGB'ye dönüştürün.",
    action="store_true",
)
    # --random_crop, --random_erasing, --convert_to_RGB, --pretrained_weights: Veri dönüşümleri ve önceden eğitilmiş ağırlıklar gibi çeşitli seçenekleri belirtir
    # Önceden Eğitilmiş Ağırlıklar, --pretrained_weights argümanı, model ile önceden eğitilmiş ağırlıkları kullanmayı belirtir

parser.add_argument(
    "--pretrained_weights",
    help="Model ile mümkünse önceden eğitilmiş ağırlıkları kullanın.",
    action="store_true",
)

    # Yukarıda tanımlanan tüm argümanları komut satırından okunur ve args değişkenine atanır

args = parser.parse_args()

    # Sinir ağı modelini (SimpleCNNModel veya ResNet18) eğitilir, doğruluğu kontrol edilir ve test edilir
    # Veriler yüklenir, model oluşturulur, eğitim ve test işlemleri gerçekleştirilir ve sonuçlar kaydedilir

def main():
    # --- KURULUM ---
    torch.backends.cudnn.benchmark = True

    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.manual_seed_all(args.manual_seed)

    # Deney adı oluşturun
    if not args.experiment_name:
        current_datetime = datetime.datetime.now()
        timestamp = current_datetime.strftime("%y%m%d-%H%M%S")
        args.experiment_name = f"{timestamp}_{args.model}"

    # Tüm deney sonuçlarının saklanacağı klasörü oluşturun
    args.experiment_path = os.path.join("experiments", args.experiment_name)
    if not os.path.isdir(args.experiment_path):
        os.makedirs(args.experiment_path)

    # Sınıfları içe aktarın
    with open(args.path_classes) as json_file:
        classes = json.load(json_file)

    # --- VERİ ---
    # Dönüşümleri oluşturun
    train_list_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]

    test_list_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]

    # Dönüşüm listesine rastgele kırpma ekleyin
    if args.random_crop:
        train_list_transforms.insert(0, transforms.RandomCrop(28, padding=4))

    # Dönüşüm listesine rastgele silme ekleyin
    if args.random_erasing:
        train_list_transforms.append(
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value="random",
                inplace=False,
            )
        )

    if args.convert_to_RGB:
        convert_to_RGB = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        train_list_transforms.append(convert_to_RGB)
        test_list_transforms.append(convert_to_RGB)

    # Eğitim Verileri
    train_transform = transforms.Compose(train_list_transforms)

    train_dataset = datasets.FashionMNIST(
        root="data", train=True, transform=train_transform, download=True
    )

    # Eğitim seti ve doğrulama seti boyutunu tanımlayalım
    train_set_length = int(
        len(train_dataset) * (100 - args.percentage_validation_set) / 100
    )
    val_set_length = int(len(train_dataset) - train_set_length)
    train_set, val_set = torch.utils.data.random_split(
        train_dataset, (train_set_length, val_set_length)
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    )

    # Test Verileri
    test_transform = transforms.Compose(test_list_transforms)

    test_set = datasets.FashionMNIST(
        root="./data", train=False, transform=test_transform, download=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # --- MODEL ---
    if args.model == "SimpleCNNModel":
        model = SimpleCNNModel()

    elif args.model == "ResNet18":
        model = models.resnet18(pretrained=args.pretrained_weights)
        model.fc = nn.Linear(model.fc.in_features, len(classes))

    num_trainable_parameters = utils.count_parameters(
        model, only_trainable_parameters=True
    )

    # Modeli GPU'ya yükleyin (mevcutsa)
    if use_cuda:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )

    # Eğitimden önce model ve parametreler hakkında bilgi yazdırın
    print(
        f"{num_trainable_parameters} eğitilebilir parametre ile {args.model} yüklendi (GPU: {use_cuda})."
    )
    print(args)

    # --- MODEL EĞİTİMİ & TEST ---
    start_num_iteration = 0
    start_epoch = 0
    best_accuracy = 0.0
    epochs_without_improvement = 0
    purge_step = None

    # Eğitimde kullanılacak başlangıç değerleri ayarlanır ve mevcutsa son kontrol noktası geri yüklenir
    checkpoint_filepath = os.path.join(args.experiment_path, "checkpoint.pth.tar")
    if os.path.exists(checkpoint_filepath):
        print(f"{checkpoint_filepath} konumundan son kontrol noktasını geri yüklüyor...")
        checkpoint = torch.load(checkpoint_filepath)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        start_num_iteration = checkpoint["num_iteration"] + 1
        best_accuracy = checkpoint["best_accuracy"]
        purge_step = start_num_iteration
        print(
            f"Son kontrol noktası geri yüklendi. {start_epoch + 1}. epochta {100 * best_accuracy:05.3f} en iyi doğruluk ile başlıyor."
        )

    # Eğitim ve doğrulama adımları için tensorboard özet yazarlarını oluşturun
    train_writer = SummaryWriter(
        os.path.join(args.experiment_path, "train"), purge_step=purge_step
    )
    valid_writer = SummaryWriter(
        os.path.join(args.experiment_path, "valid"), purge_step=purge_step
    )
    test_writer = SummaryWriter(
        os.path.join(args.experiment_path, "test"), purge_step=purge_step
    )

    for epoch in range(start_epoch, args.num_epochs):
        start_time = time.time()
        (
            train_loss,
            train_accuracy,
            num_iteration,
            batch_iteration,
        ) = utils.train(
            model=model,
            data_loader=BackgroundGenerator(train_loader),
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
            num_epoch=args.num_epochs,
            use_cuda=use_cuda,
            num_iteration=start_num_iteration,
            summary_writer=train_writer,
        )

        val_loss, val_accuracy = utils.valid(
            model=model,
            data_loader=BackgroundGenerator(val_loader),
            criterion=criterion,
            epoch=epoch,
            num_epoch=args.num_epochs,
            use_cuda=use_cuda,
            num_iteration=num_iteration,
            summary_writer=valid_writer,
        )

        utils.write_on_board(
            model=model,
            summary_writer=train_writer,
            loss=train_loss,
            accuracy=train_accuracy,
            epoch=epoch,
            num_iteration=num_iteration,
            mode="Train",
        )
        utils.write_on_board(
            model=model,
            summary_writer=valid_writer,
            loss=val_loss,
            accuracy=val_accuracy,
            epoch=epoch,
            num_iteration=num_iteration,
            mode="Validation",
        )

        # En iyi model parametrelerini kaydedin
        is_best = val_accuracy > best_accuracy
        best_accuracy = max(val_accuracy, best_accuracy)

        utils.save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_accuracy": best_accuracy,
                "optimizer_state_dict": optimizer.state_dict(),
                "num_iteration": num_iteration,
            },
            is_best,
            experiment_path=args.experiment_path,
        )

        if is_best:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement > args.patience:
                print(f"Erken durdurma, {args.patience} epoktan fazla iyileşme olmadan {epoch + 1}. epokta gerçekleşti.")
                break

        # Her epoch için zamanlayıcıyı yeniden başlatın
        start_num_iteration = num_iteration

        print(
            f"{epoch + 1}/{args.num_epochs} epoch [{int(time.time() - start_time)} saniye] - Eğitim kaybı: {train_loss:05.3f} - Eğitim doğruluğu: {train_accuracy:05.3f} - Doğrulama kaybı: {val_loss:05.3f} - Doğrulama doğruluğu: {val_accuracy:05.3f} - En iyi doğruluk: {best_accuracy:05.3f}"
        )

    utils.test(
        model=model,
        data_loader=BackgroundGenerator(test_loader),
        criterion=criterion,
        use_cuda=use_cuda,
        summary_writer=test_writer,
    )


if __name__ == "__main__":
    main()
