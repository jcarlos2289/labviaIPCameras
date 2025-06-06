import cv2
import mediapipe as mp
import threading
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2 # Necesario para dibujar los landmarks de la nueva API

# --- Clase para manejar el flujo de una cámara en un hilo separado (MediaPipe Tasks API) ---
class CameraThreadTasks:
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

        # Configuración del detector de pose para este hilo (MediaPipe Tasks API)
        base_options = python.BaseOptions(
            model_asset_path=self.model_path,
            delegate=python.BaseOptions.Delegate.GPU # ¡Aquí se especifica la GPU!
        )
        
        # Opciones específicas para PoseLandmarker
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            min_pose_detection_confidence=0.5, # Nombre correcto para la detección de pose en Tasks API
            min_tracking_confidence=0.5       # Este nombre se mantiene para el tracking de landmarks
        )

        self.detector = vision.PoseLandmarker.create_from_options(options)

        self.frame = None
        self.ret = False
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True 

        # Utilidades de dibujo (aún usamos mp.solutions.drawing_utils para dibujar)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose # Necesario para POSE_CONNECTIONS

        print(f"Cámara '{camera_source}' inicializada para ventana '{window_name}'.")
        print(f"Intentando usar GPU con MediaPipe Tasks API para cámara '{camera_source}'.")

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

            # Realizar la detección con el nuevo detector
            detection_result = self.detector.detect(mp_image)

            # Dibujar los landmarks si se detectaron
            if detection_result.pose_landmarks:
                for pose_landmarks in detection_result.pose_landmarks:
                    # Convertir los landmarks a un formato que mp_drawing pueda entender
                    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    pose_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in pose_landmarks
                    ])
                    self.mp_drawing.draw_landmarks(
                        frame,
                        pose_landmarks_proto,
                        self.mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                    )
            
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
        self.detector.close() # Cierre del detector de la nueva API
        self.thread.join()

---

## Función Principal (Manejo de 4 Cámaras con GPU Intentada)

```python
def deteccion_cuatro_camaras_gpu():
    # El modelo de MediaPipe Pose Landmarker. Se descargará automáticamente.
    model_path = 'pose_landmarker.task' 

    # --- ¡CONFIGURA AQUÍ LAS FUENTES DE TUS 4 CÁMARAS! ---
    # Puedes mezclar IDs de cámaras USB y URLs de cámaras IP.
    # Si no tienes 4 cámaras físicas, puedes repetir el ID de una cámara USB existente para probar.
    
    # EJEMPLOS:
    # cam_sources = [0, 1, 2, 3] # Para 4 webcams USB
    # cam_sources = [
    #     'rtsp://user:pass@192.168.1.100:554/stream',
    #     'rtsp://user:pass@192.168.1.101:554/stream',
    #     0, # Una webcam USB
    #     '[http://192.168.1.102:8080/video](http://192.168.1.102:8080/video)'
    # ]

    # Configuración por defecto para probar (cambia a tus cámaras reales)
    cam_sources = [0, 1, 2, 3] 

    output_scale_factor = 0.5 

    camera_threads = []
    for i, source in enumerate(cam_sources):
        thread = CameraThreadTasks( # Usamos la nueva clase CameraThreadTasks
            camera_source=source, 
            window_name=f"Camara {i+1}: Deteccion de Pose (GPU intentado)",
            model_path=model_path,
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
    print("Recuerda: MediaPipe Pose en Python actualmente tiene un problema conocido con el uso de GPU.")

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
    deteccion_cuatro_camaras_gpu()