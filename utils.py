import os
import shutil

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn

    # utils.py
    # Bu PyTorch tabanlı bir derin öğrenme projesindeki kullanılabilecek yardımcı işlevler ve sınıflar içerir

def count_parameters(model: nn.Module, only_trainable_parameters: bool = False,) -> int:
    """ Modeldeki parametre sayısını sayar

    Args:
        model (nn.Module): Parametrelerin sayılacağı model.
        only_trainable_parameters (bool: False): Sadece eğitilebilir parametreleri say.

    Returns:
        num_parameters (int): Modeldeki parametre sayısı.
    """
    if only_trainable_parameters:
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        num_parameters = sum(p.numel() for p in model.parameters())

    return num_parameters


def save_checkpoint(
    current_epoch: int,
    num_iteration: int,
    best_accuracy: float,
    model_state_dict: dict,
    optimizer_state_dict: dict,
    is_best: bool,
    experiment_path: str,
    checkpoint_filename: str = "checkpoint.pth.tar",
    best_filename: str = "model_best.pth.tar",
):
    """ Kontrol noktasını ve en iyi modeli diske kaydeder

    Args:
        current_epoch (int): Eğitimin mevcut epok sayısı.
        num_iteration (int): Eğitimin başlangıcından itibaren geçen iterasyon sayısı.
        best_accuracy (float): Eğitim sırasında elde edilen son en iyi doğruluk.
        model_state_dict (dict): Modelin durumu hakkında bilgiler içeren sözlük.
        optimizer_state_dict (dict): Optimizer'ın durumu hakkında bilgiler içeren sözlük.
        is_best (bool): Mevcut modeli yeni en iyi model olarak kaydetmek için boolean.
        experiment_path (str): Kontrol noktalarını ve en iyi modeli kaydetmek için dizin yolu.
        checkpoint_filename (str: "checkpoint.pth.tar"): Kontrol noktasına verilecek dosya adı.
        best_filename (str: "model_best.pth.tar"): En iyi modelin kontrol noktasına verilecek dosya adı.

    """
    print(
        f'Kontrol noktası kaydediliyor{f" ve yeni en iyi model (en iyi doğruluk: {100 * best_accuracy:05.2f})" if is_best else f""}...'
    )
    checkpoint_filepath = os.path.join(experiment_path, checkpoint_filename)
    torch.save(
        {
            "epoch": current_epoch,
            "num_iteration": num_iteration,
            "best_accuracy": best_accuracy,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
        },
        checkpoint_filepath,
    )
    if is_best:
        shutil.copyfile(
            checkpoint_filepath, os.path.join(experiment_path, best_filename),
        )


class MetricTracker:
    """Bir metriğin ortalama ve mevcut değerini hesaplar ve saklar."""

    def __init__(self):
        self.reset()

    def reset(self):
        """ Tüm takip edilen parametreleri sıfırlar """
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value: float, num: int = 1):
        """ Takip edilen parametreleri günceller

        Args:
            value (float): Takipçinin güncelleneceği yeni değer
            num (int: 1): Değerin hesaplanmasında kullanılan öğe sayısı
        """
        self.value = value
        self.sum += value
        self.count += num
        self.average = self.sum / self.count


class ConfusionMatrix:
    """Bir karışıklık matrisini saklar, günceller ve çizer."""

    def __init__(self, classes: dict):
        """ Bir karışıklık matrisi oluşturur ve başlatır

        Args:
            classes (dict): Tüm sınıfları içeren bir sözlük (ör. {"0": "label_0", "1": "label_1",...})
        """
        self.classes = classes
        self.num_classes = len(self.classes)
        self.labels_classes = range(self.num_classes)
        self.list_classes = list(self.classes.values())

        self.cm = np.zeros([len(classes), len(classes)], dtype=np.int)

    def update_confusion_matrix(self, targets: torch.Tensor, predictions: torch.Tensor):
        """ Karışıklık matrisini günceller

        Args:
            targets (torch.Tensor): CPU üzerindeki hedef sınıfları içeren tensör
            predictions(torch.Tensor): CPU üzerindeki tahmin edilen sınıfları içeren tensör
        """
        # Karışıklık matrisini güncellemek için sklearn kullanılır
        self.cm += confusion_matrix(targets, predictions, labels=self.labels_classes)

    def plot_confusion_matrix(
        self,
        normalize: bool = True,
        title: str = None,
        cmap: matplotlib.colors.Colormap = plt.cm.Blues,
    ) -> matplotlib.figure.Figure:
        """
        Bu fonksiyon karışıklık matrisini çizer.

        Args:
            normalize (bool: True): Karışıklık matrisinin normalizasyonunu kontrol eden boolean.
            title (str: ""): Figür için başlık
            cmap (matplotlib.colors.Colormap: plt.cm.Blues): Renk haritası, varsayılan olarak 'Blues'

        Returns:
            matplotlib.figure.Figure: Gösterilmeye/hazırlanmaya hazır olan figür
        """
        if not title:
            title = f"Normalleştirilmiş Karışıklık Matrisi" if normalize else f"Karışıklık Matrisi"

        if normalize:
            self.cm = self.cm.astype("float") / np.maximum(
                self.cm.sum(axis=1, keepdims=True), 1
            )

        # Sınıfların sayısına bağlı olarak figür oluşturulur.
        fig, ax = plt.subplots(
            figsize=[0.4 * self.num_classes + 4, 0.4 * self.num_classes + 2]
        )
        im = ax.imshow(self.cm, interpolation="nearest", cmap=cmap)
        ax.figure.colorbar(im, ax=ax)

        # Tüm işaretlerin gösterilmesi ve ilgili listelerle etiketlenmesi
        # Şeklin kesilmesini önlemek için başlangıçta ve sonda bir işaret eklenir.
        ax.set(
            xticks=np.arange(-1, self.cm.shape[1] + 1),
            yticks=np.arange(-1, self.cm.shape[0] + 1),
            xticklabels=[""] + self.list_classes,
            yticklabels=[""] + self.list_classes,
            title=title,
            ylabel="Gerçek etiket",
            xlabel="Tahmin edilen etiket",
        )

        # İşaretlerin döndürülmesi ve hizalanması
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Veri boyutları üzerinde döngü yapılır ve metin notları oluşturulur
        fmt = ".2f" if normalize else "d"
        thresh = self.cm.max() / 2.0
        for i in range(self.cm.shape[0]):
            for j in range(self.cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(self.cm[i, j], fmt),
                    ha="center",
                    va="center",
                    color="white" if self.cm[i, j] > thresh else "black",
                )
        fig.tight_layout()

        return fig
