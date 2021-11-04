import cv2
import os
import imutils

person1 = 'Persona 1'
path = './train'
full_path = path + '/' + person1

if not os.path.exists(full_path):
	print('Carpeta creada: ',full_path)
	os.makedirs(full_path)

capturas = cv2.VideoCapture(0) #Capturar rostros de mi camara

face_classif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml') #Clasificador de OpenCV para detectar rostros
count = 1

while True:
	ret, frame = capturas.read() #Leer cada frame del video
	if ret == False: break
	frame =  imutils.resize(frame, width=640) #Redimensionar los frames del video para que no sean muy grandes
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	aux_frame = frame.copy()

	faces = face_classif.detectMultiScale(gray,1.3,5)

	for (x,y,w,h) in faces: #For donde guardamos las imagenes que obtiene de nuestra camara. La resolucion que se le puso es de 150x150 pixeles.
		cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		rostro = aux_frame[y:y+h,x:x+w]
		rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
		cv2.imwrite(full_path + '/rotro_{}.jpg'.format(count),rostro)
		count = count + 1
	cv2.imshow('frame',frame) #Mostrar una ventana con las imagenes que captura de nuestra camara

	k =  cv2.waitKey(1)
	if k == 27 or count > 300: #Solo permitir capturar 300 imagenes
		break

capturas.release() #Cerramos la camara para que deje de capturar
cv2.destroyAllWindows() #Destruimos la ventana de la camara