import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger l'image
image = cv2.imread('image.png')  # Remplace par ton image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Appliquer un seuil pour segmenter les objets non noirs
_, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

# Trouver les contours des objets
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Copier les images pour affichage
image_centers = image.copy()
image_masked = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convertir le masque en image couleur

# Liste des centres des objets
centers = []

for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:  # Éviter division par zéro
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        centers.append((cx, cy))

        # Dessiner le centre en rouge 🔴
        cv2.circle(image_centers, (cx, cy), 5, (0, 0, 255), -1)  
        
        # Générer 4 points aléatoires à l'intérieur de l'objet
        random_points = []
        for _ in range(4):
            while True:
                rand_x = np.random.randint(cx - 15, cx + 15)
                rand_y = np.random.randint(cy - 15, cy + 15)

                # Vérifier que le point aléatoire est bien à l'intérieur de l'objet
                if cv2.pointPolygonTest(contour, (rand_x, rand_y), False) >= 0:
                    random_points.append((rand_x, rand_y))
                    break  # Sortir de la boucle une fois qu'un point valide est trouvé

        # Affichage des points aléatoires en bleu 🔵
        for rx, ry in random_points:
            cv2.circle(image_centers, (rx, ry), 5, (255, 0, 0), -1)  
            cv2.circle(image_masked, (rx, ry), 5, (255, 0, 0), -1)  

# Afficher les images
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].imshow(cv2.cvtColor(image_centers, cv2.COLOR_BGR2RGB))
axs[0].set_title("Image avec centres (rouge) et 4 points aléatoires (bleu)")
axs[0].axis('off')

axs[1].imshow(cv2.cvtColor(image_masked, cv2.COLOR_BGR2RGB))
axs[1].set_title("Masque avec points aléatoires (bleu)")
axs[1].axis('off')

plt.show()
