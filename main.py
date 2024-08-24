import cv2
import matplotlib.pyplot as plt
from dataset_review import label_list
from image_processing_utils import crop_bbox, segment_image

# Örnek kullanım
image_filename = label_list[0][0]  # istedigimiz resim dosyasının adını alıyoruz
bboxes = label_list[0][1]  # Bu resme ait bounding box koordinatlarını alıyoruz

# Plakayı croplama
cropped_images = crop_bbox(image_filename, bboxes, show=True)

# Eğer plakayı başarılı bir şekilde cropladıysak, segmentasyon yapalım
if cropped_images:
    image = cropped_images[0]  # İlk kesilen resmi alıyoruz
    segment_image(image, image_filename)
else:
    print("Kesilen resimler listesi boş!")




