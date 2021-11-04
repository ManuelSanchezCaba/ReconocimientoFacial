import cv2
import os

dataPath = './train' #Obtenemos los nombres de los directorios de los rostros
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)

#Seleccionamos el modelo que elegimos anteriormente en el entrenamiento
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Leemos el modelo que creamos en el entrenamiento
#face_recognizer.read('modeloEigenFace.xml')
#face_recognizer.read('modeloFisherFace.xml')
face_recognizer.read('modeloLBPH.xml')

capturas = cv2.VideoCapture(0) #Hacemos uso de la captura de nuestra camara para hacer el test del modelo entrenado

face_classif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml') #Clasificador de OpenCV para detectar rostros

while True:
	ret,frame = capturas.read()
	if ret == False: break
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	aux_frame = gray.copy()

	faces = face_classif.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces: #Redimensionamos las imagenes a como lo habiamos hecho en la captura de datos.
		rostro = aux_frame[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(150,150),interpolation= cv2.INTER_CUBIC)
		result = face_recognizer.predict(rostro) #Esta linea hace la prediccion retornandonos un label. Este label debe ser el que haga un match con el rostro.
		'''
		# EigenFaces
		if result[1] < 5700:
			cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		else:
			cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
		
		# FisherFace
		if result[1] < 500:
			cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		else:
			cv2.putText(frame,'Desconocido',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
		'''
		# LBPHFace
		if result[1] < 70: #Esta linea es para mostrar un rectangulo en el rostro del test mostrandonos el nombre de esta.
			cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x,y-25), 2, 1.1, (0,255,0), 1, cv2.LINE_AA)
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
		else:
			cv2.putText(frame, 'Desconocido', (x,y-20), 2, 0.8, (0,0,255), 1, cv2.LINE_AA)
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
		
	cv2.imshow('frame', frame)
	k = cv2.waitKey(1)
	if k == 27:
		break

capturas.release() #Cerramos la camara para que deje de capturar
cv2.destroyAllWindows() #Destruimos la ventana de la camara