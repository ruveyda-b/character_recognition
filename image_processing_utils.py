import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from dataset_review import label_list


def crop_bbox(image_filename, bboxes, image_dir="images", show=False):
    """
    Bir görüntüdeki plaka bölgesini kesen metot.
    
    Args:
        image_filename (str): Resim dosyasının adı.
        bboxes (list): Plaka için bounding box koordinatları (xmin, ymin, xmax, ymax).
        image_dir (str): Resimlerin bulunduğu klasör yolu.
        show (bool): Kesilmiş görüntüyü gösterip göstermemek için.
    
    Returns:
        cropped_images (list): Kesilmiş plaka görüntülerinin listesi.
    """
    image_path = os.path.join(image_dir, image_filename)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Resim yüklenemedi: {image_path}")
        return None
    
    cropped_images = []
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = bbox
        cropped_image = image[ymin:ymax, xmin:xmax]
        cropped_images.append(cropped_image)

        if show:
            plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            plt.title(f"Cropped bbox from {image_filename}")
            plt.show()
    
    return cropped_images

def segment_image(image, image_filename):
    h, w = image.shape[:2]  # Orijinal resim boyutları
    image_resized = cv2.resize(image, (w * 2, h * 2))   # Resmi 2 kat büyütme
    h, w = image_resized.shape[:2]  # Büyütülen resmin yeni boyutlarını alıyoruz
    plaka_alani = h * w # Plaka alanı
    
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    
    th_image = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    kernel = np.ones((3, 3), np.uint8)
    th_image_clean = cv2.morphologyEx(th_image, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(th_image_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]  # İlk 15 kontur

    print(f"Total contours found: {len(contours)}")
    
    minx_contour_list = []

    for contour in contours:
        rectangle = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rectangle)
        box = np.int64(box)
        
        # Dikdörtgenin genişliği ve yüksekliği
        (x, y), (rect_w, rect_h), angle = rectangle

        # Kontur alanını hesapla
        kontur_alani = rect_w * rect_h

        # Filtreleme koşulları - plaka boyutuna oranlama
        oran_min = 0.01  # %1
        oran_max = 0.30  # %30

        check1 = oran_min * plaka_alani < kontur_alani < oran_max * plaka_alani
        check2 = (rect_h > h * 0.03 and rect_h < h * 0.7) and (rect_w > w * 0.03 and rect_w < w * 0.7)
        
        if check1 and check2:
            minx = np.min(box[:, 0])
            minx_contour_list.append((minx, contour, box))

    # minx değerine göre sıralama
    minx_contour_list = sorted(minx_contour_list, key=lambda x: x[0])

    # Sıralanan konturları işleme ve kaydetme
    for i, (minx, contour, box) in enumerate(minx_contour_list):
        minx = np.min(box[:, 0])
        miny = np.min(box[:, 1])
        maxx = np.max(box[:, 0])
        maxy = np.max(box[:, 1])

        padding = 2  # Karakterleri tam köşelerinden aldığı için dikdörtgenlerin padding'ini 2 piksel arttırıyoruz

        minx = max(0, minx - padding)
        miny = max(0, miny - padding)
        maxx = min(image_resized.shape[1], maxx + padding)
        maxy = min(image_resized.shape[0], maxy + padding)

        # Yeni box oluşturma (padding uygulanmış)
        padded_box = np.array([[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy]])

        # Konturu dikdörtgen olarak çizdirme
        image_copy = image_resized.copy()
        cv2.drawContours(image_copy, [padded_box], 0, (0, 255, 0), 2)
        
        # # Kesme işlemi, verisetine noise sinifi olusturmak icin yazdigim kod
        # cut = image_resized[miny:maxy, minx:maxx].copy()

        # # Kesilen görüntüyü kaydetme
        # output_dir = "characters"
        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)

        # output_path = os.path.join(output_dir, f"{os.path.splitext(image_filename)[0]}_contour_{i}.jpg")
        # cv2.imwrite(output_path, cut)

        # # Kesilen bölgeyi gösterme
        # plt.imshow(cv2.cvtColor(cut, cv2.COLOR_BGR2RGB))
        # plt.title(f"Cropped Contour {i} from {image_filename}")
        # plt.show()

        # Kesilen bölgeyi gösterme yerine tüm konturlar sıralandıktan sonra plaka üzerinde gösteriliyor
        plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
        plt.title(f"Contour {i} on {image_filename}")
        plt.show()




