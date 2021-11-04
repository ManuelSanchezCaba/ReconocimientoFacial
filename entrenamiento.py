import cv2
import os
import numpy as np

"""
Obtenemos el listado de las carpetas con los rostros de las personas
"""
path = './train'
people_list = os.listdir(path)
print('Lista de personas: ', people_list)

labels = []
faces_data = []
label = 0

"""
Asignar a cada imagen de las personas que tengamos un label que nos ayudara
a determinar si es una persona 'x' o si es desconocida. Por ejemplo, si tenemos dos carpetas
de personas 1 y 2, a la persona 1 se identificara con el label 0 y la 2 con el label 1, asi poder identificar
que persona es cada quien.
"""

for name_direct in people_list:
    person_path = path + '/' + name_direct
    print('Leyendo las imágenes')
    
    for file_name in os.listdir(person_path):
        print('Rostros: ', name_direct + '/' + file_name)
        labels.append(label)
        faces_data.append(cv2.imread(person_path + '/' + file_name, 0))
        image = cv2.imread(person_path + '/' + file_name, 0) #Estas lineas son para que nos muestre las imagenes que esta guardando en escalas de grises.
        cv2.imshow('image', image)
        cv2.waitKey(10)
    label = label + 1

# Seleccionamos el metodo que queremos usar para entrenar
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Entrenamos nuestra AI con el modelo seleccionado y con la data de los rostros junto con el label asignado a cada imagen
print("Entrenando...")
face_recognizer.train(faces_data, np.array(labels))

# Almacenando el modelo obtenido
#face_recognizer.write('modeloEigen.xml')
#face_recognizer.write('modeloFisher.xml')
face_recognizer.write('modeloLBPH.xml')
print("Modelo almacenado...")