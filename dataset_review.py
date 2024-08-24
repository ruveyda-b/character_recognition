import os
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt

# Resimlerin ve etiketlerin bulunduğu dizinler
image_dir = "images"  # Resimlerin bulunduğu klasör yolu
label_dir = "labels"  # Etiketlerin (XML dosyalarının) bulunduğu klasör yolu

# Plaka koordinatlarını ve resim adlarını tutacak liste
label_list = []

# labels.txt dosyasını aç (eğer yoksa oluştur)
with open("labels.txt", "w") as label_file:
    
    # Resimlerin ve ilgili XML dosyalarının işlenmesi
    for image_file in os.listdir(image_dir):
        if image_file.endswith('.jpg'):
            # Resim dosyasının tam yolu
            image_path = os.path.join(image_dir, image_file)
            image = cv2.imread(image_path)  # Resmi yükle

            if image is None:
                print(f"Resim yüklenemedi: {image_path}")
                continue

            # İlgili XML dosyasını bulma
            xml_file = os.path.splitext(image_file)[0] + ".xml"
            xml_path = os.path.join(label_dir, xml_file)

            if not os.path.exists(xml_path):
                print(f"İlgili XML dosyası bulunamadı: {xml_path}")
                continue

            # XML dosyasını işleyerek bounding box koordinatlarını al
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Koordinatları tutacak liste
            bboxes = []

            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')
                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                # Bounding box'ı koordinatlar listesine ekle
                bboxes.append((xmin, ymin, xmax, ymax))
                
                # Bounding box'ı görüntüde gösterme (isteğe bağlı)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
            # Label listesine bu resim ve plaka koordinatlarını ekle
            label_list.append([image_file, bboxes])

            # labels.txt dosyasına yazma
            label_file.write(f"Resim: {image_file}\n")
            for bbox in bboxes:
                label_file.write(f"Koordinatlar: {bbox}\n")
            label_file.write("\n")  # Her resim için bir boş satır ekleyelim

            # Resmi yeniden boyutlandır ve göster (isteğe bağlı)
            image = cv2.resize(image, (500, 500))
            # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # plt.title(image_file)
            # plt.show()

# bu adıma kadar fotograflara bbox lari cizdirdik ve koordinatları da labels[] listesine aldik





    
 