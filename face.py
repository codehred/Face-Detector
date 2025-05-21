import cv2

# Nos aseguramos que opencv esta correctamente instalado y la versión que tenemos
print(cv2.__version__)

# Variable que se encargará de almacenar lo que vea la cámara
captura = cv2.VideoCapture(0)

# La variable "detector_de_rostros" contendrá el modelo que me ayudará
# a detectar si la cámara esta viendo una cara o no.
detector_de_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

detector_de_ojos = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Cargando el clasificador para sonrisas
detector_de_sonrisas = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

while True:
  # Esto almacenará lo siguiente:
  # - retenido: Un valor booleano que nos ayuda a determinar si la foto se tomó de manera adecuada
  # - frame: Es un valor que almacena la información de la imagen
  retenido, frame = captura.read()


  # Creamos una variable que contiene la imagen de la cámara en blenco y negro
  imagen_byc = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


  # Hacemos la detección de mi rostro con la imagen b y n y esa detección la almacenamos en la variable
  # "mi_cara"

  # La función detectMultiScale espera recibir por lo menos 3 parámetros
  # 1.- La imagen en b y n donde realizará la detección
  # 2.- Espera recibir un factor de escala para el video
  # 3.- Un número de vecinos mínimos para hacer la detección   
  mi_cara = detector_de_rostros.detectMultiScale(imagen_byc, 1.1, 5)

  # Ese factor de escala 1.1 le está indicando que vamos a reducir la imagen en un 10% 
  # El número de vecinos se refiere a cuantros "recuadros" se tiene que considerar
  # para ser detectada como un rostro



  for (x, y, largo_rostro, alto_rostro) in mi_cara:
  
    cv2.rectangle(frame, (x, y), (x+largo_rostro, y+alto_rostro), (255,0,0), 2)


    print(f''' 
      posicion en x: {x}
      posicion en y: {y}
      largo rostro (px): {largo_rostro}
      altura rostro (px): {alto_rostro}
    ''')
    # Recortamos el rostro de la imagen original
    seccion_cara = frame[y:y+alto_rostro, x:x+largo_rostro]

    # Recortamos el rostro de la imagen en blanco y negro
    seccion_cara_byn = imagen_byc[y:y+alto_rostro, x:x+largo_rostro]
    # Creamos un rectangulo para mi rostro
    # 1.- La imagen donde se va a dibujar el rectangulo
    # 2.- Los puntos de inicio para dibujar el rectangulo
    # 3.- Los puntos finales para dibujar el rectangulo
    # 4.- El color del rectangulo
    mis_ojos = detector_de_ojos.detectMultiScale(seccion_cara_byn, 1.1, 20)

    mi_sonrisa = detector_de_sonrisas.detectMultiScale(seccion_cara_byn, 1.4, 30)


  
    for (x_ojos, y_ojos, largo_ojos, alto_ojos) in mis_ojos:
      cv2.rectangle(frame, (x + x_ojos, y + y_ojos), 
                  (x + x_ojos + largo_ojos, y + y_ojos + alto_ojos), 
                  (0,255,0), 1)

    # Dibujar la sonrisa
    for (x_sonrisa, y_sonrisa, largo_sonrisa, alto_sonrisa) in mi_sonrisa:
      cv2.rectangle(frame, (x + x_sonrisa, y + y_sonrisa), 
                  (x + x_sonrisa + largo_sonrisa, y + y_sonrisa + alto_sonrisa), 
                  (0,0,255), 1)


  # cv to Color

  # imshow -> image show: Reconstruye esos pixeles (frames) en una imagen
    cv2.imshow('FACE DETECTOR', frame)

  # Agregamos una forma de salir del programa
  if cv2.waitKey(1) & 0xFF == ord('q'):
    # Instrucción 
    break

cv2.destroyAllWindows()