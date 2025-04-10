AutoBallSystem/
├── main.py
├── ArduinoControl.ino
├── requirements.txt
├── coco.names
├── yolov3.cfg
├── yolov3.weights
├── README.md
[![https://drive.google.com/file/d/15-TiLdq4T1wI6TTEvQePOeVmlD2D16u_/view?usp=drivesdk](https://img.youtube.com/vi/ID_الفيديو/0.jpg)](https://youtu.be/ID_الفيديو)
import cv2
import numpy as np
import serial
import time

# الاتصال بـ Arduino عبر منفذ USB (تأكد من تعديل المنفذ حسب نظامك، مثل "/dev/ttyUSB0" أو "COM3")
try:
    ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
    time.sleep(2)
    print("تم الاتصال مع Arduino.")
except Exception as e:
    print("فشل الاتصال بـ Arduino:", e)
    ser = None

def send_launch_command():
    if ser:
        ser.write(b"LAUNCH\n")
        print("تم إرسال أمر LAUNCH إلى Arduino.")

# تحميل نموذج YOLO والملفات المرتبطة به
try:
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
except Exception as e:
    print("فشل تحميل ملفات النموذج:", e)
    exit(1)

# قراءة أسماء الفئات من coco.names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# الحصول على طبقات الإخراج في النموذج
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# نركز على فئة "sports ball" للكشف عنها
target_label = "sports ball"

# فتح الكاميرا
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("فشل تشغيل الكاميرا.")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("فشل التقاط الإطار.")
        break

    height, width = frame.shape[:2]
    # تجهيز الصورة لإدخال النموذج (blob)
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    detected = False
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == target_label:
                detected = True
                # حساب أبعاد وإحداثيات المستطيل
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, target_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # إذا لم يتم اكتشاف الكرة يتم إرسال أمر الإطلاق
    if not detected:
        print("لم يتم اكتشاف الكرة، جاري إرسال أمر الإطلاق.")
        send_launch_command()

    cv2.imshow("Ball Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
    // تعريف دبوس الإطلاق (يمكن توصيله بمخرج تحكم في المحرك، مثل رلية أو Driver)
const int launchPin = 8;

// تعريف دبابيس حساس المسافة HC‑SR04
const int trigPin = 9;
const int echoPin = 10;

// عتبة المسافة (بـ سم)؛ إذا كانت المسافة أكبر من هذه القيمة، نفترض غياب الكرة في غرفة الإطلاق
const long ballThreshold = 20;  // يمكنك تعديل هذه القيمة حسب التجربة

void setup() {
  Serial.begin(9600);
  
  // إعداد دبابيس الإطلاق
  pinMode(launchPin, OUTPUT);
  digitalWrite(launchPin, LOW);
  
  // إعداد دبابيس حساس المسافة
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  
  Serial.println("نظام Arduino جاهز لاستقبال الأوامر.");
}

// دالة لقياس المسافة بواسطة HC‑SR04
long readDistance() {
  // التأكد من تنظيف دبوس التريغ
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  // إرسال نبضة قصيرة إلى حساس المسافة
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  // قراءة مدة النبضة على دبوس الصدى
  long duration = pulseIn(echoPin, HIGH);
  // تحويل المدة إلى مسافة (speed of sound ≈ 34300 سم/ثانية)
  long distance = duration * 0.034 / 2;
  return distance;
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    Serial.print("الأمر المستلم: ");
    Serial.println(command);
    
    if (command == "LAUNCH") {
      // قياس المسافة للتحقق من وجود أو عدم وجود الكرة في غرفة الإطلاق
      long distance = readDistance();
      Serial.print("المسافة المقاسة: ");
      Serial.print(distance);
      Serial.println(" سم");
      
      // إذا كانت المسافة أكبر من العتبة، نعتبر غرفة الإطلاق فارغة وننفذ عملية الإطلاق
      if (distance > ballThreshold) {
        Serial.println("غرفة الإطلاق فارغة، جاري تنفيذ الإطلاق.");
        digitalWrite(launchPin, HIGH);
        delay(500);  // مدة تشغيل المحرك (يمكن تعديلها حسب آلية الإطلاق)
        digitalWrite(launchPin, LOW);
        Serial.println("تم تنفيذ الإطلاق.");
      }
      else {
        Serial.println("الكرة موجودة في غرفة الإطلاق، لن يتم تفعيل الإطلاق.");
      }
    }
  }
}
ser = serial.Serial('COM3', 9600)  # تغيير إلى المنفذ الصحيح 
python main.py
opencv-python
numpy
pyserial
person
bicycle
car
motorbike
aeroplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
sofa
pottedplant
bed
diningtable
toilet
tvmonitor
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush

[net]
# Testing
# batch=1
# subdivisions=1
# Training
batch=64
subdivisions=16
width=608
height=608
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.001
burn_in=1000
max_batches = 500200
policy=steps
steps=400000,450000
scales=.1,.1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky

# Downsample

[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=128
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=256
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=512
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

######################

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear


[yolo]
mask = 6,7,8
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1


[route]
layers = -4

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 61



[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear


[yolo]
mask = 3,4,5
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1



[route]
layers = -4

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 36



[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear


[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
[net]
# Testing
# batch=1
# subdivisions=1
# Training
batch=64
subdivisions=16
width=608
height=608
channels=3
momentum=0.9
decay=0.0005
angle=0
saturation = 1.5
exposure = 1.5
hue=.1

learning_rate=0.001
burn_in=1000
max_batches = 500200
policy=steps
steps=400000,450000
scales=.1,.1

[convolutional]
batch_normalize=1
filters=32
size=3
stride=1
pad=1
activation=leaky
# Downsample

[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=128
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=256
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=512
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

######################

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear


[yolo]
mask = 6,7,8
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1


[route]
layers = -4

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 61



[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear


[yolo]
mask = 3,4,5
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1



[route]
layers = -4

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 36



[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=255
activation=linear


[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=80
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=1
https://pjreddie.com/media/files/yolov3.weights الملف الثقيل yolov3.weights 

# نظام التعبئة الآلي في الملاعب

يهدف هذا المشروع إلى تطوير نظام آلي ذكي لاستبدال حاملي الكرات في الملاعب باستخدام تقنيات الذكاء الاصطناعي، وذلك عبر:
- **Raspberry Pi** لمعالجة الفيديو والكشف عن الكرات باستخدام OpenCV وخوارزمية YOLO.
- **Arduino** للتحكم في آلية الإطلاق عبر المحرك، مع دمج حساس المسافة (HC‑SR04) للتأكد من خلو غرفة الإطلاق من الكرة قبل تنفيذ الأمر.

## مكونات النظام:
- **Raspberry Pi**: لتشغيل كود Python، معالجة الفيديو، وتشغيل نموذج YOLO.
- **Arduino**: لتلقي أوامر الإطلاق من Raspberry Pi والتحكم بالمحرك.
- **كاميرا**: لالتقاط الفيديو.
- **محرك/آلية إطلاق**: يتم التحكم به عبر Arduino (ممكن تشغيله عن طريق رلية أو Driver).
- **حساس مسافة HC‑SR04**: لقياس المسافة والتأكد من عدم وجود كرة في غرفة الإطلاق قبل تفعيل الإطلاق.

## محتويات المستودع:
- `main.py`: كود Raspberry Pi الذي يقوم بالتقاط الفيديو والكشف عن كرات الرياضة، ثم إرسال أمر "LAUNCH" في حالة عدم اكتشاف الكرة.
- `ArduinoControl.ino`: كود Arduino الذي يستقبل أمر "LAUNCH"، يقيس المسافة باستخدام حساس HC‑SR04، وينفذ عملية الإطلاق إذا كانت غرفة الإطلاق فارغة.
- `requirements.txt`: المكتبات المطلوبة لتشغيل كود Python.
- `coco.names`: ملف أسماء الفئات لنموذج YOLO.
- `yolov3.cfg` و `yolov3.weights`: ملفات نموذج YOLO.
- `README.md`: هذا الملف للتوثيق.

## كيفية التشغيل:
1. **على Raspberry Pi:**
   - ثبت المكتبات اللازمة:
     ```bash
     pip install -r requirements.txt
     ```
   - تأكد من توصيل الكاميرا بشكل صحيح.
   - قم بتعديل منفذ الاتصال التسلسلي في `main.py` إذا لزم الأمر (مثلاً تغيير `/dev/ttyUSB0` إلى المنفذ المناسب).
   - شغل الكود:
     ```bash
     python main.py
     ```

2. **على Arduino:**
   - حمّل الكود الموجود في `ArduinoControl.ino` إلى لوحة Arduino.
   - تأكد من توصيل:
     - دبوس الإطلاق (مثلاً رقم 8) بالمحرك أو وحدة الإطلاق.
     - حسّاس HC‑SR04:  
       - `trigPin` على الدبوس 9،  
       - `echoPin` على الدبوس 10.
   - قم بتعديل قيم العتبة (ballThreshold) إذا لزم الأمر للتحقق من غياب الكرة في غرفة الإطلاق.

## ملاحظات:
- تأكد من وضع ملفات النموذج (`yolov3.cfg`، `yolov3.weights`، `coco.names`) في نفس مجلد المشروع.
- تأكد من توافق توصيلات الـ Arduino مع الكود (الأرقام المستخدمة للدبابيس).
- الكود مُصمم للتنبيه والإطلاق في حالة عدم وجود كرة في غرفة الإطلاق وفق قراءة حساس المسافة.
- يمكنك تطوير الكود وإضافة أي تحسينات بناءً على متطلبات المشروع الفعلية.

---

بهذا يكون المشروع متكاملاً من جهة معالجة الصورة والتحكم الآلي، ويحتوي على كافة الأكواد والحساسات اللازمة دون أخطاء مع شروحات مفصلة لكل جزء من الكود.

# ReturnX-Master-The-Field
مشروع نظام التعبئة الآلي للملاعب باستخدام YOLO وArduino
# نظام التعبئة الآلي في الملاعب

## وصف المشروع
يهدف هذا المشروع إلى استبدال نظام حاملي الكرات البشريين بنظام آلي ذكي يستخدم تقنيات الذكاء الاصطناعي والتعلم العميق. يعتمد المشروع على معالجة الفيديو والصورة باستخدام Raspberry Pi مع OpenCV وخوارزمية YOLO للكشف عن الكرات، وفي حالة عدم اكتشاف الكرة يتم إرسال أمر "LAUNCH" إلى لوحة Arduino التي تتحكم في آلية الإطلاق لإطلاق كرة جديدة. كما تم إضافة حساس مسافة (HC‑SR04) للتأكد من خلو غرفة الإطلاق قبل تفعيل النظام، مما يضمن دقة أكبر ويقلل من حدوث أخطاء.

## مكونات النظام
- **Raspberry Pi**: لمعالجة الفيديو والصورة وتشغيل نموذج YOLO باستخدام Python.
- **كاميرا**: لالتقاط الفيديو من الملعب.
- **Arduino**: لاستقبال أوامر الإطلاق عبر الاتصال التسلسلي والتحكم بالمحرك.
- **محرك/آلية إطلاق**: يتم التحكم فيه عبر Arduino لتنفيذ عملية الإطلاق.
- **حساس مسافة HC‑SR04**: لقياس المسافة والتأكد من غياب الكرة في غرفة الإطلاق قبل تفعيل النظام.
- **ملفات النموذج الخاصة بـ YOLO**:
  - `yolov3.cfg`
  - `yolov3.weights`
  - `coco.names` (تأكد من احتوائه على فئة "sports ball" أو ما يناسب هدفك)

## طريقة التشغيل
1. **إعداد Raspberry Pi:**
   - تأكد من تثبيت المكتبات المطلوبة باستخدام ملف `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```
   - يجب وضع ملفات النموذج (yolov3.cfg، yolov3.weights، و coco.names) في نفس المجلد الذي يوجد به ملف `main.py`.
   - قم بتوصيل الكاميرا بالـ Raspberry Pi.

2. **تشغيل كود الـ Python (main.py):**
   - يقوم الكود بالتقاط الفيديو، معالجة الصور باستخدام نموذج YOLO، والتأكد من وجود كرة رياضية في الإطار.
   - في حالة عدم اكتشاف الكرة، يقوم الكود بإرسال أمر "LAUNCH" إلى Arduino عبر منفذ USB.

3. **إعداد Arduino:**
   - حمل الكود الموجود في ملف `ArduinoControl.ino` إلى لوحة Arduino.
   - تأكد من توصيل دبوس التحكم (مثلاً الدبوس رقم 8) بالمحرك أو آلية الإطلاق.
   - قم بتوصيل حساس المسافة HC‑SR04 إلى الدبابيس المحددة (مثلاً: trig على الدبوس 9 و echo على الدبوس 10).
   - يتأكد الكود من قياس المسافة، وإذا كانت الغرفة فارغة (أي أنه لا توجد كرة قريبة حسب العتبة المحددة)، يقوم Arduino بتفعيل نظام الإطلاق.

## ملاحظات إضافية
- **الاتصال التسلسلي:** تأكد من تعديل منفذ الاتصال التسلسلي في ملف `main.py` (مثل `/dev/ttyUSB0` أو `COM3`) بما يتناسب مع نظام التشغيل الذي تستخدمه.
- **ضبط عتبة حساس المسافة:** يمكنك تعديل قيمة العتبة (ballThreshold) في الكود الخاص بـ Arduino بحيث تناسب ظروف التشغيل الفعلية.
- **توسيع النموذج:** يمكنك لاحقاً تعديل ملف `coco.names` أو استخدام نموذج YOLO مخصص إذا أردت تحسين دقة الكشف لتناسب متطلبات المشروع بشكل أفضل.
- **التوثيق والتحديث:** يُنصح بتحديث هذا الملف دوريًا مع أي تغييرات أو تحسينات في المشروع.

---

يُظهر هذا المشروع التكامُل بين معالجة الصور الذكية باستخدام تقنيات الذكاء الاصطناعي والتحكم الآلي عبر Arduino، مما يجعله حلاً مبتكرًا لتحسين أداء الملاعب وإدارة الكرة بشكل آلي.
