import cv2
import os
import imutils
import numpy as np
import tkinter

class captura_datos:
    def __init__(self, path):
        self.count_persona = len(os.listdir(path)) + 1
        self.count = 1

    def inicializar_carpeta_rostro(self, name, path):
        full_path = path + '/' + name
        if not os.path.exists(full_path):
            print('Carpeta creada: ',full_path)
            os.makedirs(full_path)

    def guardar_rostros(self, frame, face_classif, full_path):
        frame =  imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aux_frame = frame.copy()

        faces = face_classif.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
            rostro = aux_frame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(full_path + '/rotro_{}.jpg'.format(self.count), rostro)
            self.count += 1

        cv2.imshow('frame',frame)

class entrenamiento:
    def __init__(self):
        self.labels = []
        self.faces_data = []
        self.label = 0

    def asignar_label_rostros(self, path, people_list):
        for name_direct in people_list:
            person_path = path + '/' + name_direct
            print('Leyendo las imágenes')
            
            for file_name in os.listdir(person_path):
                self.labels.append(self.label)
                self.faces_data.append(cv2.imread(person_path + '/' + file_name, 0))
                image = cv2.imread(person_path + '/' + file_name, 0) #Estas lineas son para que nos muestre las imagenes que esta guardando en escalas de grises.
                cv2.imshow('image', image)
                cv2.waitKey(10)
            self.label = self.label + 1

    def generar_modelo(self, modelo):
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        print('Entrenando...')
        face_recognizer.train(self.faces_data, np.array(self.labels))
        face_recognizer.write(modelo)
        cv2.destroyAllWindows()
        print('Termino...')

class reconocimiento_facial:
    def __init__(self, path):
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.imagePaths = os.listdir(path)

    def utilizar_modelo(self, modelo):
        self.face_recognizer.read(modelo)

    def detectar_rostro(self, frame, face_classif):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aux_frame = gray.copy()

        faces = face_classif.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces: #Redimensionamos las imagenes a como lo habiamos hecho en la captura de datos.
            rostro = aux_frame[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
            result = self.face_recognizer.predict(rostro) #Esta linea hace la prediccion retornandonos un label. Este label debe ser el que haga un match con el rostro.

            if result[1] < 70: #Esta linea es para mostrar un rectangulo en el rostro del test mostrandonos el nombre de esta.
                cv2.putText(frame, '{}'.format(self.imagePaths[result[0]]), (x,y-25), 2, 1.1, (0,255,0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Desconocido', (x,y-20), 2, 0.8, (0,0,255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)

def detener_captura(capturas):
    capturas.release() #Cerramos la camara para que deje de capturar
    cv2.destroyAllWindows() #Destruimos la ventana de la camara

cd = captura_datos('./train')

def capturar_datos():
    capturas = cv2.VideoCapture(0) #Capturar rostros de mi camara
    face_classif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml') #Clasificador de OpenCV para detectar rostros
    
    cd.inicializar_carpeta_rostro('persona' + str(cd.count_persona), './train')

    while True:
        ret, frame = capturas.read() #Leer cada frame del video
        if ret == False: return

        if cd.count > 300:
            detener_captura(capturas)
            cd.count_persona = cd.count_persona + 1
            cd.count = 1
        else:
            cd.guardar_rostros(frame, face_classif, './train/persona' + str(cd.count_persona))
            cv2.waitKey(1)

def generar_modelo():
    train = entrenamiento()
    train.asignar_label_rostros('./train', os.listdir('./train'))
    train.generar_modelo('modeloLBPH.xml')

def reconocedor_rostros():
    capturas = cv2.VideoCapture(0) #Capturar rostros de mi camara
    face_classif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml') #Clasificador de OpenCV para detectar rostros
    fr = reconocimiento_facial('./train')
    fr.utilizar_modelo('modeloLBPH.xml')

    while True:
        ret, frame = capturas.read() #Leer cada frame del video
        if ret == False: return

        fr.detectar_rostro(frame, face_classif)
        cv2.imshow('frame', frame)

        k = cv2.waitKey(1)
        if k == 27: #Salir si se le da a esc
            break
    
    detener_captura(capturas)

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