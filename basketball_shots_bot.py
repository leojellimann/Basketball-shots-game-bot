# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 19:57:34 2023

@author: Léo
"""

import numpy as np
from PIL import ImageGrab
import pyautogui
import time
import keyboard  # Importez la bibliothèque keyboard
from threading import Thread, Lock
import cv2
import pandas as pd
import asyncio

# Définissez les coordonnées et la taille de la région de l'écran à capturer
x = 1240  # Coordonnée x du coin supérieur gauche
y = 270  # Coordonnée y du coin supérieur gauche
largeur = 400  # Largeur de la région à capturer
hauteur = 215  # Hauteur de la région à capturer

# Charger l'image du panier de basket comme modèle
basket_template = cv2.imread('bucketgray.png', 0)  # Assurez-vous que l'image est en niveau de gris

# Initialiser l'ORB détecteur
orb = cv2.ORB_create()

# Trouver les points clés et les descripteurs avec ORB dans le modèle
kp1, des1 = orb.detectAndCompute(basket_template, None)

# Définir le critère de correspondance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Initialiser une liste pour stocker les positions
positions = []

def detect_circle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=30, maxRadius=50)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            return (i[0], i[1], i[2])  # Retourne les coordonnées (x, y) du premier cercle trouvé
    return None

def find_basket_in_frame(frame, orb, kp1, des1, bf):
    # Convertir le cadre capturé en niveau de gris
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Trouver les points clés et les descripteurs dans le cadre
    kp2, des2 = orb.detectAndCompute(gray_frame, None)

    # S'assurer qu'il y a des descripteurs à apparier
    if des2 is not None:
        # Trouver les correspondances des descripteurs
        matches = bf.match(des1, des2)

        # Trier les correspondances dans l'ordre de leur distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # Supposer qu'il y a suffisamment de correspondances pour continuer
        if len(matches) > 10:
            # Extraire les emplacements des points correspondants
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Calculer l'homographie entre les points de modèle et les points de l'image capturée
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # S'assurer que l'homographie a été trouvée
            if M is not None:
                # Déterminer les points du rectangle autour du panier dans l'image de modèle
                h, w = basket_template.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

                # Projeter les points du rectangle dans l'image capturée
                dst = cv2.perspectiveTransform(pts, M)

                # Calculer le centre du panier détecté
                center = np.mean(dst, axis=0).reshape(-1)
                basket_center = (int(center[0] + x), int(center[1] + y))
                return basket_center
    return None

def bucket_to_right(bucket_center, bucket_center0, circle_coords):
    if bucket_center and bucket_center[0] > bucket_center0[0]: #si le panier va vers la droite
        print(f"le panier est présent en position: x:{bucket_center[0]} et y{bucket_center[1]}")
        pyautogui.moveTo(circle_coords[0], circle_coords[1], duration=0.1)
        pyautogui.mouseDown()
        time.sleep(0.1)

        """                             BALLE A DROITE DU PANIER/PANIER SE RAPPROCHE                                    """

        #si la balle est à droite du panier mais que le panier est à moins de 50px de la balle
        if circle_coords[0] - bucket_center[0] > 0 and circle_coords[0] - bucket_center[0] <= 50:#tir droit si le panier arrive à hauteur de la balle
            pyautogui.moveTo(circle_coords[0], circle_coords[1]-100, duration=0.1)#envoyer la balle vers haut tout droit
            pyautogui.mouseUp()
        #si la balle est à droite du panier mais que le panier est entre 50 et 100px de la balle
        elif circle_coords[0] - bucket_center[0] > 0 and circle_coords[0] - bucket_center[0] > 50 and circle_coords[0] - bucket_center[0] <= 100:
            pyautogui.moveTo(bucket_center[0]+50, circle_coords[1]-100, duration=0.1)#envoyer la balle vers haut tout droit
            pyautogui.mouseUp()
        #si la balle est à droite du panier mais que le panier est entre 100 et 150px de la balle
        elif circle_coords[0] - bucket_center[0] > 0 and circle_coords[0] - bucket_center[0] > 100 and circle_coords[0] - bucket_center[0] <= 150:
            pyautogui.moveTo(bucket_center[0]+100, circle_coords[1]-100, duration=0.1)#envoyer la balle vers haut tout droit
            pyautogui.mouseUp()
        #si la balle est à droite du panier mais que le panier est entre 150 et 200px de la balle
        elif circle_coords[0] - bucket_center[0] > 0 and circle_coords[0] - bucket_center[0] > 150 and circle_coords[0] - bucket_center[0] <= 200:
            pyautogui.moveTo(bucket_center[0]+150, circle_coords[1]-100, duration=0.1)#envoyer la balle vers haut tout droit
            pyautogui.mouseUp()
        #si la balle est à droite du panier mais que le panier est entre 50 et 150px de la balle
        elif circle_coords[0] - bucket_center[0] > 0 and circle_coords[0] - bucket_center[0] > 200 and circle_coords[0] - bucket_center[0] <= 250:
            pyautogui.moveTo(bucket_center[0]+175, circle_coords[1]-100, duration=0.1)#envoyer la balle vers haut tout droit
            pyautogui.mouseUp()
        #si la balle est à droite du panier mais que le panier est à plus de 150px de la balle
        elif circle_coords[0] - bucket_center[0] > 0 and circle_coords[0] - bucket_center[0] > 250 :
            pyautogui.moveTo(bucket_center[0]+200, circle_coords[1]-100, duration=0.1)#envoyer la balle vers haut tout droit
            pyautogui.mouseUp()

        #"""                             BALLE A GAUCHE DU PANIER/PANIER S'ELOIGNE                                    """
        #si la balle est à gauche du panier mais que le panier est à moins de 50px de la balle
        elif circle_coords[0] - bucket_center[0] < 0 and abs(circle_coords[0] - bucket_center[0]) <= 50:#tir droit si le panier arrive à hauteur de la balle
            pyautogui.moveTo(circle_coords[0]+50, circle_coords[1]-100, duration=0.1)#envoyer la balle vers haut tout droit
            pyautogui.mouseUp()
        #si la balle est à gauche du panier mais que le panier est entre 50 et 300px de la balle
        elif circle_coords[0] - bucket_center[0] < 0 and abs(circle_coords[0] - bucket_center[0]) > 50 and abs(circle_coords[0] - bucket_center[0]) <= 100:
            pyautogui.moveTo(bucket_center[0]+100, circle_coords[1]-100, duration=0.1)#envoyer la balle vers haut tout droit
            pyautogui.mouseUp()
        #si la balle est à gauche du panier mais que le panier est entre 50 et 300px de la balle
        elif circle_coords[0] - bucket_center[0] < 0 and abs(circle_coords[0] - bucket_center[0]) > 100 and abs(circle_coords[0] - bucket_center[0]) <= 150:
            pyautogui.moveTo(bucket_center[0]+150, circle_coords[1]-100, duration=0.1)#envoyer la balle vers haut tout droit
            pyautogui.mouseUp()
        #si la balle est à gauche du panier mais que le panier est à plus de 300px de la balle
        elif circle_coords[0] - bucket_center[0] < 0 and abs(circle_coords[0] - bucket_center[0]) > 150 :
            pyautogui.moveTo(bucket_center[0]+175, circle_coords[1]-100, duration=0.1)#envoyer la balle vers haut tout droit
            pyautogui.mouseUp()

        screenshot = pyautogui.screenshot(region=(x, y, largeur, hauteur))
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bucket_center = find_basket_in_frame(frame, orb, kp1, des1, bf)#nouvelle position
        print(f"après le tir, le panier est en position: x:{bucket_center[0]} et y{bucket_center[1]}")
        time.sleep(0.4)
        print("fin de l'attente")

        return True

def bucket_to_left(bucket_center, bucket_center0, circle_coords):
    if bucket_center and bucket_center[0] < bucket_center0[0]: #si le panier va vers la gauche
        print(f"le panier est présent en position: x:{bucket_center[0]} et y{bucket_center[1]}")
        pyautogui.moveTo(circle_coords[0], circle_coords[1], duration=0.1)
        pyautogui.mouseDown()
        time.sleep(0.1)

        #si la balle est à droite du panier mais que le panier est à moins de 50px de la balle
        if circle_coords[0] - bucket_center[0] > 0 and circle_coords[0] - bucket_center[0] <= 50:#tir droit si le panier arrive à hauteur de la balle
            pyautogui.moveTo(circle_coords[0]-75, circle_coords[1]-100, duration=0.1)#envoyer la balle vers haut tout droit
            pyautogui.mouseUp()
        #si la balle est à droite du panier mais que le panier est entre 50 et 150px de la balle
        elif circle_coords[0] - bucket_center[0] > 0 and circle_coords[0] - bucket_center[0] > 50 and circle_coords[0] - bucket_center[0] <= 150:
            pyautogui.moveTo(bucket_center[0]-100, circle_coords[1]-100, duration=0.1)#envoyer la balle vers haut tout droit
            pyautogui.mouseUp()
        #si la balle est à droite du panier mais que le panier est à plus de 150px de la balle
        elif circle_coords[0] - bucket_center[0] > 0 and circle_coords[0] - bucket_center[0] > 150 :
            pyautogui.moveTo(bucket_center[0]-150, circle_coords[1]-100, duration=0.1)#envoyer la balle vers haut tout droit
            pyautogui.mouseUp()

        #si la balle est à gauche du panier mais que le panier est à moins de 50px de la balle
        elif circle_coords[0] - bucket_center[0] < 0 and abs(circle_coords[0] - bucket_center[0]) <= 50:#tir droit si le panier arrive à hauteur de la balle
            pyautogui.moveTo(circle_coords[0], circle_coords[1]-100, duration=0.1)#envoyer la balle vers haut tout droit
            pyautogui.mouseUp()
        #si la balle est à gauche du panier mais que le panier est entre 50 et 300px de la balle
        elif circle_coords[0] - bucket_center[0] < 0 and abs(circle_coords[0] - bucket_center[0]) > 50 and abs(circle_coords[0] - bucket_center[0]) <= 150:
            pyautogui.moveTo(bucket_center[0]-100, circle_coords[1]-100, duration=0.1)#envoyer la balle vers haut tout droit
            pyautogui.mouseUp()
        #si la balle est à gauche du panier mais que le panier est à plus de 300px de la balle
        elif circle_coords[0] - bucket_center[0] < 0 and abs(circle_coords[0] - bucket_center[0]) > 150 :
            pyautogui.moveTo(bucket_center[0]-150, circle_coords[1]-100, duration=0.1)#envoyer la balle vers haut tout droit
            pyautogui.mouseUp()

        screenshot = pyautogui.screenshot(region=(x, y, largeur, hauteur))
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bucket_center = find_basket_in_frame(frame, orb, kp1, des1, bf)#nouvelle position
        print(f"après le tir, le panier est en position: x:{bucket_center[0]} et y{bucket_center[1]}")
        time.sleep(0.4)
        print("fin de l'attente")
        return True





def new_bucket_to_right(bucket_center, bucket_center0, circle_coords):
    pyautogui.moveTo(circle_coords[0], circle_coords[1], duration=0.1)
    pyautogui.mouseDown()
    time.sleep(0.1)
    #panier va à droite et est pas sur les extremités
    if bucket_center[0] < 1500 and bucket_center[0] > 1300:
        pyautogui.moveTo(bucket_center[0]+90, circle_coords[1]-100)
        pyautogui.mouseUp()
        print("balle envoyée vers la droite")
        time.sleep(1)
        return True
    #panier va à droite et est sur l'extremité droite
    elif bucket_center[0] >= 1500 and bucket_center[0] > 1360:
        pyautogui.moveTo(1500, circle_coords[1]-100)
        pyautogui.mouseUp()
        print("balle envoyée vers l'ex droite")
        time.sleep(1)
        return True
    else:
        print("C'est la merde right")
        return True

def new_bucket_to_left(bucket_center, bucket_center0, circle_coords):
    pyautogui.moveTo(circle_coords[0], circle_coords[1], duration=0.1)
    pyautogui.mouseDown()
    time.sleep(0.1)
    #panier va à gauche et est pas sur les extremités
    if bucket_center[0] < 1600 and bucket_center[0] > 1360:
        pyautogui.moveTo(bucket_center[0]-90, circle_coords[1]-100)
        pyautogui.mouseUp()
        print("balle envoyée vers la gauche")
        time.sleep(1)
        return True
    #panier va à gauche et est sur l'extremité gauche
    elif bucket_center[0] < 1600 and bucket_center[0] <= 1360:
        pyautogui.moveTo(1360, circle_coords[1]-100)
        pyautogui.mouseUp()
        print("balle envoyée vers l'ex gauche")
        time.sleep(1)
        return True
    else:
        print("C'est la merde left")
        return True




def main():
    prev = (0, 0, 0) #previous position of the ball
    #circle_coords = (None, None, None)#current position of the ball
    bucket_center0 = (None, None)#previous postion of the bucket
    bucket_center = (None, None)#current positon of the bucket
    process_complete = True
    try:
        while True:
            #print(f"le panier est en position x: {bucket_center[0]} y: {bucket_center[1]}")
            # Vérifier si bucket_center n'est pas None
            #if bucket_center is not None:
                #print(f"le panier est en position x: {bucket_center[0]} y: {bucket_center[1]}")
                #positions.append({'x': bucket_center[0], 'y': bucket_center[1]})


            #if keyboard.is_pressed('space'):
                #break
        # Créer un DataFrame pandas à partir de la liste des positions
        #df = pd.DataFrame(positions)

        # Exporter le DataFrame dans un fichier Excel
        #df.to_excel('positions_panier.xlsx', index=False)

            if process_complete:
                should_continue = True
                while should_continue:
                    screen = np.array(ImageGrab.grab(bbox=None))
                    circle_coords = detect_circle(screen)
                    if circle_coords is not None:
                        #if abs(prev[0]-circle_coords[0]) < 10 and abs(prev[1] - circle_coords[1]) < 100:
                        if abs(prev[0]-circle_coords[0]) < 70 and circle_coords[1] > 760 and circle_coords[1] <805:
                            print(f"le ballon est bien placé car {prev} == {circle_coords}")
                            process_complete = False
                            should_continue = False
                        else:
                            print(f"le ballon est pas bien placé car {prev} != {circle_coords}")
                            prev = circle_coords
                            time.sleep(0.1)
                    else:
                        print("Aucune balle détectée correctement au démarrage.")
                        time.sleep(0.1)
                print("le balon a été detecté, je vais le lancer")
            #print("je suis sorti du while")
            if circle_coords and not process_complete:
                # Capturer une capture d'écran de la région spécifiée
                screenshot0 = pyautogui.screenshot(region=(x, y, largeur, hauteur))
                frame0 = np.array(screenshot0)
                frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
                bucket_center0 = find_basket_in_frame(frame0, orb, kp1, des1, bf)#précédente position
                screenshot = pyautogui.screenshot(region=(x, y, largeur, hauteur))
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bucket_center = find_basket_in_frame(frame, orb, kp1, des1, bf)#nouvelle position

                #process_complete = bucket_to_right(bucket_center, bucket_center0, circle_coords)
                #process_complete = bucket_to_left(bucket_center, bucket_center0, circle_coords)

                if bucket_center and bucket_center[0] > bucket_center0[0]:
                    process_complete = new_bucket_to_right(bucket_center, bucket_center0, circle_coords)
                elif bucket_center and bucket_center[0] < bucket_center0[0]:
                    process_complete = new_bucket_to_left(bucket_center, bucket_center0, circle_coords)
                elif bucket_center and (abs(bucket_center[0] - bucket_center0[0]) <= 50 or bucket_center[0] == bucket_center0[0]): #si le panier est centré avec la balle
                #refaire un test screenshot etc
                    #print(f"le panier est présent en position: x:{bucket_center[0]} et y{bucket_center[1]}")
                    print("CAS MILIEU !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print(f"Ancienne position panier x: {bucket_center0[0]}, vs nouvelle x: {bucket_center[0]}")
                    print("CAS MILIEU !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    pyautogui.moveTo(circle_coords[0], circle_coords[1], duration=0.1)
                    pyautogui.mouseDown()
                    time.sleep(0.1)
                    #panier va à droite et est pas sur les extremités
                    if bucket_center[0] < 1500 and bucket_center[0] > 1360:
                        pyautogui.moveTo(bucket_center[0], circle_coords[1]-100)
                        pyautogui.mouseUp()
                        print("j'envoie au milieu milieu")

                    elif bucket_center[0] >= 1500 :
                        pyautogui.moveTo(1500, circle_coords[1]-100)
                        pyautogui.mouseUp()
                        print("j'envoie au milieu ex droite")
                    elif bucket_center[0] <= 1360 :
                        pyautogui.moveTo(1360, circle_coords[1]-100)
                        pyautogui.mouseUp()
                        print("j'envoie au milieu ex gauche")
                    else:
                        print("C'est le milieu")
                    """
                    if bucket_center and bucket_center[0] > bucket_center0[0] and bucket_center[0] < 1500 and bucket_center[0] > 1360:
                        pyautogui.moveTo(bucket_center[0]+90, circle_coords[1]-100)
                        pyautogui.mouseUp()
                        print("balle envoyée vers la droite après être centré")
                    #panier va à gauche et est pas sur les extremités
                    elif bucket_center and bucket_center[0] < bucket_center0[0] and bucket_center[0] < 1500 and bucket_center[0] > 1360:
                        pyautogui.moveTo(bucket_center[0]-90, circle_coords[1]-100)
                        pyautogui.mouseUp()
                        print("balle envoyée vers la gauche après être centré")
                    #panier va à droite et est sur l'extremité droite
                    elif bucket_center and bucket_center[0] > bucket_center0[0] and bucket_center[0] >= 1500 and bucket_center[0] > 1360:
                        pyautogui.moveTo(1500, circle_coords[1]-100)
                        pyautogui.mouseUp()
                        print("balle envoyée vers l'ex droite après être centré")
                    #panier va à gauche et est sur l'extremité gauche
                    elif bucket_center and bucket_center[0] < bucket_center0[0] and bucket_center[0] < 1500 and bucket_center[0] <= 1360:
                        pyautogui.moveTo(1360, circle_coords[1]-100)
                        pyautogui.mouseUp()
                        print("balle envoyée vers l'ex gauche après être centré")
                    else:
                        print("C'est la merde milieu")
                    """

                    #screenshot = pyautogui.screenshot(region=(x, y, largeur, hauteur))
                    #frame = np.array(screenshot)
                    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    #bucket_center = find_basket_in_frame(frame, orb, kp1, des1, bf)#nouvelle position
                    #print(f"après le tir, le panier est en position: x:{bucket_center[0]} et y{bucket_center[1]}")
                    process_complete = True
                    time.sleep(1)
                else:
                    print("pas de solution trouvée")

                # Pause rapide pour ne pas surcharger le CPU
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            else:
                print("Aucune balle trouvée")
                # Attendre 500 ms

    except KeyboardInterrupt:
        print("Programme interrompu par l'utilisateur.")

    # Fermer les fenêtres OpenCV et nettoyer
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
