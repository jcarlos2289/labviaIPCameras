import cv2
import mediapipe as mp
import threading
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# No necesitamos landmark_pb2 para FaceDetector, ya que dibuja bounding boxes y keypoints directamente.
# from mediapipe.framework.formats import landmark_pb2 

# --- Clase para manejar el flujo de una cámara en un hilo separado (MediaPipe Tasks API) ---
class CameraThreadFaceDetection:
    def __init__(self, camera_source, window_name, model_path, output_size_factor=0.5):
        self.camera_source = camera_source
        self.window_name = window_name
        self.model_path = model_path
        self.output_size_factor = output_size_factor 
        
        self.cap = cv2.VideoCapture(camera_source) 
        
        if not self.cap.isOpened():
            print(f"Error: No se pudo abrir la cámara {camera_source}. Asegúrate de que la URL o ID sea correcta y accesible.")
            self.cap = None
            return

        # Configuración del detector facial para este hilo (MediaPipe Tasks API)
        base_options = python.BaseOptions(
            model_asset_path=self.model_path,
            delegate=python.BaseOptions.Delegate.GPU # ¡Aquí se especifica la GPU!
        )
        
        # Opciones específicas para FaceDetector
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.5, # Umbral de confianza para la detección
            # model_selection=0 # Puedes ajustar el modelo si es necesario (0: short-range, 1: full-range)
        )

        # Crear el detector facial
        self.detector = vision.FaceDetector.create_from_options(options)

        self.frame = None
        self.ret = False
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True 

        print(f"Cámara '{camera_source}' inicializada para ventana '{window_name}'.")
        print(f"Intentando usar GPU con MediaPipe Tasks API para detección facial en '{camera_source}'.")

    def _run(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print(f"Advertencia: No se pudo leer el frame de la cámara {self.camera_source}. Intentando reconectar en 3 segundos...")
                self.cap.release()
                time.sleep(3) 
                self.cap = cv2.VideoCapture(self.camera_source)
                if not self.cap.isOpened():
                    print(f"Error: No se pudo reconectar con la cámara {self.camera_source}. Deteniendo hilo.")
                    self.running = False
                continue 

            frame = cv2.flip(frame, 1)

            # Convertir a RGB y crear mp.Image para la nueva API
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.RGB, data=image_rgb)

            # Realizar la detección
            # Nota: detect_for_video() es más apropiado para streams de video y permite timestamps
            # Sin embargo, para simplicidad en este ejemplo con hilos separados, detect() sigue siendo viable.
            detection_result = self.detector.detect(mp_image)

            # Dibujar los resultados de la detección (bounding boxes y keypoints)
            if detection_result.detections:
                for detection in detection_result.detections:
                    # Dibujar el bounding box
                    bbox = detection.bounding_box
                    x_min = bbox.origin_x
                    y_min = bbox.origin_y
                    width = bbox.width
                    height = bbox.height
                    cv2.rectangle(frame, (x_min, y_min), (x_min + width, y_min + height), (0, 255, 0), 2) # Verde

                    # Dibujar los puntos clave (6 keypoints faciales)
                    # Keypoints vienen normalizados, necesitamos escalarlos
                    if detection.keypoints:
                        for kp in detection.keypoints:
                            kp_x = int(kp.x * frame.shape[1])
                            kp_y = int(kp.y * frame.shape[0])
                            cv2.circle(frame, (kp_x, kp_y), 3, (0, 0, 255), -1) # Rojo, círculo relleno
            
            # Redimensionar la imagen de salida
            if self.output_size_factor != 1.0:
                width = int(frame.shape[1] * self.output_size_factor)
                height = int(frame.shape[0] * self.output_size_factor)
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            self.frame = frame 
            self.ret = ret
            time.sleep(0.001)

    def start(self):
        if self.cap is not None:
            self.thread.start()

    def get_frame(self):
        return self.ret, self.frame

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.detector.close() 
        self.thread.join()

---

## Función Principal (Manejo de 4 Cámaras con Detección Facial GPU Intentada)

```python
def deteccion_cuatro_camaras_facial_gpu():
    # El modelo de MediaPipe Face Detector para la Tasks API. Se descargará automáticamente.
    # ASEGÚRATE DE QUE EL NOMBRE DEL ARCHIVO SEA 'face_detector.task'
    # MediaPipe lo descargará de: [https://storage.googleapis.com/mediapipe-models/face_detector/face_detector/float16/1/face_detector.task](https://storage.googleapis.com/mediapipe-models/face_detector/face_detector/float16/1/face_detector.task)
    model_path = 'face_detector.task' 

    # --- ¡CONFIGURA AQUÍ LAS FUENTES DE TUS 4 CÁMARAS! ---
    # Igual que en el ejemplo de pose, configura las URLs de tus cámaras IP o IDs de USB.
    
    # EJEMPLOS:
    # cam_sources = [0, 1, 2, 3] 
    # cam_sources = [
    #     'rtsp://user:pass@192.168.1.100:554/stream',
    #     'rtsp://user:pass@192.168.1.101:554/stream',
    #     0, 
    #     '[http://192.168.1.102:8080/video](http://192.168.1.102:8080/video)'
    # ]

    # Configuración por defecto para probar (cambia a tus cámaras reales)
    cam_sources = [0, 1, 2, 3] 

    output_scale_factor = 0.5 

    camera_threads = []
    for i, source in enumerate(cam_sources):
        thread = CameraThreadFaceDetection( 
            camera_source=source, 
            window_name=f"Camara {i+1}: Deteccion Facial (GPU intentado)",
            model_path=model_path, # Pasamos el nombre de archivo .task
            output_size_factor=output_scale_factor
        )
        if thread.cap is not None: 
            camera_threads.append(thread)
        else:
            print(f"La cámara '{source}' no pudo ser iniciada, no se añadirá al procesamiento.")

    if not camera_threads:
        print("Ninguna cámara se pudo iniciar. Saliendo.")
        return

    for thread in camera_threads:
        thread.start()

    print("\nPresiona 'q' para salir de cualquier ventana.")
    print("Recordatorio: El soporte de GPU para MediaPipe en Python puede variar.")

    while True:
        all_stopped = True 

        for thread in camera_threads:
            ret, frame = thread.get_frame()
            if ret:
                cv2.imshow(thread.window_name, frame)
                all_stopped = False 
            elif thread.running: 
                all_stopped = False

        if (cv2.waitKey(1) & 0xFF == ord('q')) or all_stopped:
            break

    for thread in camera_threads:
        thread.stop()

    cv2.destroyAllWindows()
    print("Programa finalizado.")

if __name__ == '__main__':
    deteccion_cuatro_camaras_facial_gpu()