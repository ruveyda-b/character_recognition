# Joblib kütüphanesi pipeline işlerinde hızlı sonuçlar veren bir kütüphanedir. 
# Modelimizi pkl olarak kaydetmeyi ve tekrar yüklememize imkan sunan sklearn içinde yer alan bir kütüphanedir.
import joblib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Kaydedilen modeli yükleyelim
svm_model = joblib.load('svm_model_rbf.pkl')
print("Model loaded successfully.")

# Segmentlenmiş karakterleri içeren klasörün yolu
characters_dir = "characters"

# Segmentlenmiş karakterleri tanımak için
def recognize_characters(model, characters_dir):
    for filename in os.listdir(characters_dir):
        if filename.endswith('.jpg'):
            img_path = os.path.join(characters_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Görüntüyü yeniden boyutlandır ve düzleştir
                img_resized = cv2.resize(img, (28, 28)).flatten()
                img_normalized = img_resized / 255.0  # Normalizasyon

                # Tahmin yap
                prediction = model.predict([img_normalized])

                # Sonuçları göster
                plt.imshow(img, cmap='gray')
                plt.title(f"Predicted Character: {prediction[0]}")
                plt.axis('off')
                plt.show()

# Tanıma işlemini başlat
recognize_characters(svm_model, characters_dir)
