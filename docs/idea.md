1. La idea en una frase

Una cámara montada en una máquina vigila una zona de peligro alrededor de ella. Si una persona, un vehículo o un animal entra en esa zona, el sistema lo detecta y avisa a la computadora de la máquina por CAN.

Todo corre en la Jetson, sin internet y sin servidor externo. Eso es lo que se llama edge: el procesamiento se hace en el equipo mismo, al lado del sensor, y no en la nube.

Por qué este proyecto te sirve para conseguir trabajo: los avisos de Edge AI y Embedded CV piden tres cosas que hoy no podés mostrar con un proyecto propio:

Manejar el pipeline de video (cámara, GStreamer, latencia). Era tu hueco número uno.
Algo corriendo "en producción", no un notebook que anduvo una vez.
Conectar la visión con el sistema físico. Esa parte es tu diferenciador, porque casi ningún candidato de visión sabe de CAN.
2. El sistema, pieza por pieza
 ┌─────────┐   ┌───────────┐   ┌────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐   ┌──────────┐
 │ Cámara  │──▶│ GStreamer │──▶│ Slot   │──▶│ Detector │──▶│ Tracker  │──▶│ Lógica  │──▶│ Salida   │
 │ USB 3.0 │   │ pipeline  │   │ último │   │ YOLOv8n  │   │          │   │ de zona │   │ CAN      │
 └─────────┘   └───────────┘   │ frame  │   │ TensorRT │   └──────────┘   └─────────┘   └──────────┘
                               └────────┘   └──────────┘                                     │
                                                                                     ┌───────┴───────┐
                                                                                     │ Métricas y    │
                                                                                     │ logs          │
                                                                                     └───────────────┘

Un frame es una imagen individual del video. A 30 fps llegan 30 frames por segundo. Un pipeline es una cadena de pasos donde la salida de uno es la entrada del siguiente, como una línea de montaje.

Pieza 1: la cámara (V4L2)

V4L2 ("Video for Linux 2") es la parte del kernel de Linux que maneja las cámaras. Cada cámara aparece como un archivo, /dev/video0, y por ahí se le pide el formato y se leen los frames. Ya la usaste con v4l2-ctl.

Tu código: parse_v4l2_formats lee qué formatos ofrece la cámara, y pick_format elige el mejor para la resolución y los fps que necesitás. Además, al arrancar, el programa configura los controles de la cámara (la exposición, por ejemplo), porque se pierden cada vez que se desconecta.

Pieza 2: GStreamer

GStreamer es un framework (una librería grande con su propia forma de trabajar) para procesar audio y video. Funciona encadenando elementos, que son bloques que hacen una sola cosa: uno lee la cámara, otro descomprime, otro convierte colores. Se encadenan con !, igual que una tubería:

v4l2src device=/dev/video0 ! image/jpeg,width=1280,height=720 ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink

Leído en castellano: "leé la cámara, pedile JPEG de 1280x720, descomprimilo, convertí los colores a BGR (el orden que usa OpenCV) y entregámelo en mi programa". appsink es el elemento que le pasa los frames a tu código Python.

Por qué GStreamer y no cv2.VideoCapture(0) directo: porque te deja elegir cada paso, y en la Jetson podés cambiar jpegdec (que usa la CPU) por nvv4l2decoder (que usa el hardware de video). Con OpenCV solo no tenés ese control.

Tu código: build_pipeline arma ese texto según la fuente (cámara USB, RTSP o archivo de video, que te sirve para probar sin cámara).

Pieza 3: el slot del último frame

Este es el concepto más importante de la parte de video, y el que más vas a tener que defender.

El problema: la cámara entrega 48 frames por segundo, pero el detector procesa, digamos, 30. Si guardás todos los frames en una cola para procesarlos en orden, la cola crece sin parar. Al minuto estás procesando un frame de hace varios segundos. Para un sistema de seguridad, eso es lo peor que puede pasar: la persona ya está adentro de la zona y vos estás analizando la imagen de cuando todavía estaba lejos.

La solución: guardar solo el último frame. Si llega uno nuevo y el anterior no se procesó, se pisa y se descarta. Preferís perder frames antes que ir atrasado. En GStreamer eso es appsink drop=true max-buffers=1, y en tu código es la clase LatestFrame, que además cuenta los descartes. Ese contador es una métrica: si descartás muchos, el detector no da abasto.

Pieza 4: el detector (YOLOv8n con TensorRT)

YOLOv8n es un detector de objetos: recibe una imagen y devuelve bounding boxes ("cajas delimitadoras", rectángulos alrededor de cada objeto) con su clase (persona, auto, perro) y una confianza entre 0 y 1. La "n" es de nano, la versión más chica y rápida. Viene entrenado con COCO, así que ya detecta personas, vehículos y animales sin entrenar nada.

TensorRT es la librería de NVIDIA que toma un modelo y lo optimiza para la GPU de esa placa en particular. Ya lo mediste en agosto: FP16 te dio el doble de velocidad sin cambiar las detecciones.

Alrededor del modelo hay dos pasos más:

Preproceso (letterbox): el modelo espera una imagen de 640x640. Tu cámara da 1280x720, que no es cuadrada. Si la estirás, deformás a las personas. Lo correcto es achicarla manteniendo la proporción y rellenar el sobrante con gris. Eso es letterbox.
Postproceso: el modelo devuelve cajas en coordenadas de la imagen de 640x640. Hay que volver a la imagen original, deshaciendo el relleno y la escala. Eso es scale_boxes. Además se aplica NMS (Non-Maximum Suppression), que elimina las cajas repetidas sobre el mismo objeto y deja la de mayor confianza.

El bug clásico que ya viste en la ArkVision: si letterbox rellena de una forma y scale_boxes deshace de otra, todas las cajas quedan corridas y no salta ningún error. Por eso el test de ida y vuelta es el que más vale.

Pieza 5: el tracker

El detector mira cada frame por separado y no sabe que la persona del frame 100 es la misma del frame 99. El tracker ("seguidor") le asigna a cada objeto un número que se mantiene entre frames. Ya escribiste uno en la L04 (tracker_hungaro.py), que asocia las cajas de un frame con las del anterior según cuánto se superponen.

Para qué hace falta acá:

Si el detector pierde a la persona en un solo frame, sin tracker parecería que salió y volvió a entrar, y la alarma se apagaría y prendería.
Con el tracker podés decir "la persona 7 está hace 10 frames seguidos dentro de la zona", y eso es mucho más confiable que una detección aislada.
Pieza 6: la lógica de zona

La zona es un polígono que definís sobre la imagen, por ejemplo el trapecio del piso delante de la máquina. Se guarda en un archivo de configuración, no en el código.

Qué punto de la caja se mira: el punto de los pies, o sea el centro del borde de abajo de la caja. No el centro de la caja. Pensalo así: una persona parada al lado de la zona, con la cabeza "metida" en el polígono por la perspectiva de la cámara, no está en peligro. Lo que importa es dónde pisa.

El antirrebote: para disparar, el objeto tiene que estar adentro N frames seguidos (por ejemplo 5). Para apagar, tiene que estar afuera M frames seguidos (por ejemplo 15, más que para entrar). Es una histéresis, la misma idea que el antirrebote de un pulsador o un comparador con histéresis en electrónica: evita que la salida oscile cuando el objeto está justo en el borde. Que apagar tarde más que prender es una decisión de seguridad: ante la duda, la alarma sigue prendida.

Tu código: una función point_in_polygon (si un punto está dentro de un polígono) y una pequeña máquina de estados por objeto: AFUERA → ENTRANDO → ADENTRO → SALIENDO → AFUERA.

Pieza 7: la salida por CAN

Esta es tu parte fuerte. Usás python-can sobre SocketCAN (Linux maneja el PCAN como una interfaz de red, can0).

Vos diseñás el mensaje. Una propuesta para discutir:

Un ID fijo para el estado de la zona.
Byte 0: estado (0 = libre, 1 = advertencia, 2 = objeto en la zona).
Byte 1: cantidad de objetos en la zona.
Byte 2: clase del objeto más cercano (persona, vehículo, animal).
Byte 3: un contador que se incrementa en cada mensaje.
Byte 4: un código de estado del sistema (cámara caída, modelo caído, etc.).

La decisión más importante de esta pieza, y la que más vas a lucir en una entrevista: el mensaje se manda periódicamente, por ejemplo cada 100 ms, aunque no pase nada. Ese mensaje periódico se llama heartbeat ("latido"). Si la ECU deja de recibirlo, sabe que el sistema de visión murió y lo trata como una falla. Si solo mandaras un mensaje cuando hay alguien, una cámara colgada se vería igual que una zona vacía, y eso es peligroso. La ausencia de mensajes nunca puede significar "todo bien". El contador del byte 3 sirve para lo mismo: si la ECU ve el mismo número repetido, el programa está trabado aunque el bus siga mandando. Esto lo conocés de J1939 y de las ECU de trabajo, y es exactamente lo que un ingeniero de visión puro no piensa.

Pieza 8: métricas y logs
Métricas: números que el programa publica continuamente: fps reales de la cámara, fps del detector, latencia de punta a punta, frames descartados, temperatura de la Jetson, cantidad de reconexiones de la cámara. Los recolecta Prometheus (un programa que cada tantos segundos le pide los números a tu aplicación y los guarda con la hora) y los grafica Grafana, que corre en la Raspberry Pi.
Logs: el registro de eventos, cada uno con la hora: "cámara desconectada", "cámara reconectada", "persona entró en la zona". Con eso armás el registro de fallas de la semana 4.
3. Qué significa "producción" en este proyecto

Es la diferencia entre "anduvo una vez en mi escritorio" y "puedo dejarlo andando una semana y confiar en él". En concreto:

Arranca solo. Si se corta la luz, al volver la Jetson levanta el programa sin que nadie toque nada. Eso lo hace systemd, el administrador de servicios de Linux.
Se recupera solo. Si la cámara se desconecta, el programa no muere: lo registra, manda el estado de falla por CAN y reintenta cada pocos segundos. Si el programa se cuelga, systemd lo reinicia.
Corre en Docker. Docker empaqueta el programa con todas sus dependencias en una imagen, que corre igual en cualquier Jetson. Así no depende de lo que alguien instaló a mano.
Se configura sin tocar código. La zona, los umbrales, el ID de CAN y la cámara van en un archivo de configuración.
Se puede observar. Las métricas y los logs de la pieza 8.
Tiene tests, y corren solos. GitHub Actions es un servicio de GitHub que corre tus tests cada vez que subís código. Si algo se rompe, te enterás antes de instalarlo en la placa. Eso se llama CI (Continuous Integration, integración continua).
4. Cómo se organiza el repo

En inglés, porque es público. Cada pieza del diagrama es un archivo:

hazard-zone/
├── hazard_zone/
│   ├── camera.py       piezas 1, 2 y 3: formatos, pipeline, LatestFrame
│   ├── geometry.py     letterbox, scale_boxes, point_in_polygon
│   ├── detector.py     pieza 4: carga el engine de TensorRT y detecta
│   ├── tracker.py      pieza 5: tu tracker de la L04, adaptado
│   ├── zone.py         pieza 6: la máquina de estados de la zona
│   ├── can_output.py   pieza 7: el mensaje y el heartbeat
│   ├── metrics.py      pieza 8
│   └── main.py         une todo y corre el lazo principal
├── config/
│   └── default.yaml    zona, umbrales, ID de CAN, cámara
├── tests/              un archivo de tests por módulo
├── docker/
├── systemd/
└── README.md

El reparto es el de siempre: yo te doy la consigna, las firmas de las funciones y los tests que fallan, y vos escribís el código. Las partes que dependen del hardware (TensorRT, la cámara real, el PCAN) se prueban en la Jetson. Todo lo demás (geometría, zona, armado del mensaje CAN) se prueba en la notebook sin hardware. Ese es un principio de diseño que vale la pena decir en una entrevista: la lógica se separa del hardware para poder testearla sin él.

5. El plan, semana por semana
Semana	Qué	Lo que tiene que andar el viernes
1 (7 al 10 oct)	Cámara y video	camera.py y geometry.py con los tests en verde. GStreamer corriendo en la Jetson con tu cámara, mostrando fps y descartes
2 (13 al 17 oct)	Detector y zona	YOLOv8n en TensorRT dentro del pipeline, tracker, zona con antirrebote. Un video con las cajas y la zona dibujadas
3 (20 al 24 oct)	CAN y producción	Mensaje CAN con heartbeat recibido en otra máquina (candump). Docker y systemd. Reconexión de la cámara
4 (27 al 31 oct)	Observabilidad y cierre	Prometheus y Grafana en la Pi, GitHub Actions, corriendo 24 horas seguidas con registro de fallas, README en inglés con un GIF

Antes del lunes 13 tiene que estar instalado nvidia-jetpack en la Jetson, porque la semana 2 necesita TensorRT.

6. Qué hacés esta semana, en orden
Hoy: en tu repo, crear hazard_zone/camera.py y tests/test_camera.py. Copiar las firmas y los tests del grupo A y B de L02/videopipe.py y escribir parse_v4l2_formats, pick_format y build_pipeline.
Jueves: geometry.py con letterbox y scale_boxes (el grupo C de la L02), y LatestFrame (el grupo D) en camera.py.
Viernes: en la Jetson, correr el pipeline con tu cámara y medir fps y descartes. Primer commit con todo andando.

Cuando termines el grupo A, o si te trabás más de 20 minutos, pasame el código y lo revisamos.
