import cv2
import os
import imutils
import numpy as np
import tkinter

from numpy.lib.type_check import imag

class captura_datos:
    def __init__(self, image_path, image_path_list):
        self.image_path = image_path
        self.image_path_list = image_path_list

    def inicializar_carpeta_rostro(self, name):
        if not os.path.exists(name):
            print('Creando Carpeta de Rostros')
            os.makedirs(name)

    def guardar_rostros(self, face_classif):
        count = 0

        for image_name in self.image_path_list:
            print('Image Name is', image_name)
            image = cv2.imread(self.image_path + '/' + image_name) #Leer las imágenes
            image_aux = image.copy() #Hacer una copia de la imagen encontrada. Esto porque la imagen principal se estará modificando para mostrar por pantalla.
            gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #Convertir la imagen en escala de grises.
            faces = face_classif.detectMultiScale(gray_scale, 1.1, 5) #Obtener el rostro encontrado.

            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (128, 0, 255), 2) #Obtener un rectangulo del rostro encontrado
                rostro = image_aux[y:y+h, x:x+w] #Obtener rostros de la copia del rostro
                rostro = cv2.resize(rostro, (150, 150), interpolation= cv2.INTER_CUBIC) #Redimensionar a 150x150 pixeles el rostro copia
                cv2.imwrite('Rostros/rostro_{}.jpg'.format(count), rostro) #Guardar rostro encontrado en una carpeta
                count += 1
                cv2.imshow('Imagen', image) #Mostrar la imagen por pantalla.
                cv2.imshow('Image', rostro) #Mostrar la copia que se guardará por pantalla. 
                cv2.waitKey(0) #Esperar que se teclee algo

        cv2.destroyAllWindows() #Eliminar las ventanas creadas para mostrar los recortes

class entrenamiento:
    def __init__(self):
        self.labels = [] #Arreglo donde se almacenan los label
        self.faces_data = [] #Arreglo donde se almacenan los rostros
        self.label = 0 #Contador que se ira incrementando a medida que se asigna un label

    def asignar_label_rostros(self, path, face_list):
        for name_file in face_list:
            print('Leyendo las imágenes')
            self.labels.append(self.label) #Asignar label
            self.faces_data.append(cv2.imread(path + '/' + name_file, 0)) #Guardar rostro en la misma posicion donde se almaceno el label.
            image = cv2.imread(path + '/' + name_file, 0) #Estas lineas son para que nos muestre las imagenes que esta guardando en escalas de grises.
            cv2.imshow('image', image) #Mostrar por pantalla el rostro que se esta asignando el label
            cv2.waitKey(1)
            self.label = self.label + 1 #Incrementar el label para que no se dupliquen.
                

    def generar_modelo(self, modelo):
        face_recognizer = cv2.face.LBPHFaceRecognizer_create() #Creamos el tipo de modelo a crear. En este caso crearemos un modelo LBPH.
        print('Entrenando...')
        face_recognizer.train(self.faces_data, np.array(self.labels)) #Entrenamos el modelo pasandole los rostros y sus respectivos labels.
        face_recognizer.write(modelo) #Una vez entrenado, que nos escriba en el mismo directorio el modelo.
        cv2.destroyAllWindows() #Cerramos todas las ventanas que han estado abierta
        print('Termino...')

class reconocimiento_facial:
    def __init__(self, path):
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.imagePaths = os.listdir(path)

    def utilizar_modelo(self, modelo):
        self.face_recognizer.read(modelo)

    def detectar_rostro(self, frame, face_classif):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convertir el rostro en escala de grises
        aux_frame = gray.copy() #Hacer una copia

        faces = face_classif.detectMultiScale(gray,1.3,5) #Obtener el rostro

        for (x,y,w,h) in faces: #Redimensionamos las imagenes a como lo habiamos hecho en la captura de datos.
            rostro = aux_frame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
            result = self.face_recognizer.predict(rostro) #Esta linea hace la prediccion retornandonos un label. 
                                                          #Este label debe ser el que haga un match con el rostro.

            if result[1] < 70: #Esta linea es para mostrar un rectangulo en el rostro del test mostrandonos el nombre de esta.
                cv2.putText(frame, '{}'.format(self.imagePaths[result[0]]), (x,y-25), 2, 1.1, (0,255,0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else: #En caso contrario nos muestra en pantalla un rectangulo diciendonos que es desconocido.
                cv2.putText(frame, 'Desconocido', (x,y-20), 2, 0.8, (0,0,255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
            
            cv2.imshow('Image', frame) #Mostrar en pantalla
            cv2.waitKey(0) #Esperar hasta que pulsemos una tecla
        
        cv2.destroyAllWindows() #Cerrar todas las ventanas abiertas generadas por OpenCV    

def detener_captura(capturas):
    capturas.release() #Cerramos la camara para que deje de capturar
    cv2.destroyAllWindows() #Destruimos la ventana de la camara

def capturar_datos():
    image_path = './BancoImagenes'
    image_path_list = os.listdir(image_path)
    print('Images = ', image_path_list)

    face_classif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml') #Clasificador de OpenCV para detectar rostros
    
    cd = captura_datos(image_path, image_path_list)

    cd.inicializar_carpeta_rostro('Rostros')

    cd.guardar_rostros(face_classif)

def generar_modelo():
    train = entrenamiento()
    train.asignar_label_rostros('./Rostros', os.listdir('./Rostros'))
    train.generar_modelo('modeloLBPH.xml')

def reconocedor_rostros():
    image_path = './Test/imagen_1.jpg'
    image = cv2.imread(image_path)
    face_classif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml') #Clasificador de OpenCV para detectar rostros
    fr = reconocimiento_facial('./Rostros')
    
    fr.utilizar_modelo('modeloLBPH.xml')

    fr.detectar_rostro(image, face_classif)

def main():
    ventana = tkinter.Tk()
    ventana.geometry('400x300')

    boton1 = tkinter.Button(ventana, text= 'Capturar Datos', padx= 400, pady= 20, command= capturar_datos)
    boton1.pack()

    boton2 = tkinter.Button(ventana, text= 'Generar Modelo', padx= 400, pady= 20, command= generar_modelo)
    boton2.pack()

    boton3 = tkinter.Button(ventana, text= 'Reconocer Rostro', padx= 400, pady= 20, command= reconocedor_rostros)
    boton3.pack()

    ventana.mainloop()

main()