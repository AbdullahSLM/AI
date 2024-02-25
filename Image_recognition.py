import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

# جمع البيانات
car_photos = ['sport_car.jpg', 'sport_car2.jpg', 'sport_car3.jpg', 'sport_car4.jpg', 'sport_car5.jpg', 'family_car.jpg', 'family_car2.jpg',
              'family_car3.jpg', 'family_car4.jpg', 'family_car5.jpg', 'small_car.jpg', 'small_car2.jpg', 'small_car3.jpg', 'small_car4.jpg', 'small_car5.jpg']
labels = [0, 1, 2]
data = []
count = i = int(0)
for photo in car_photos:
    count += 1
    img = Image.open(photo)
    img = img.resize((224, 224))
    img = np.array(img)
    data.append([img, labels[i]])
    if count == 5:
        i += 1
        count = 0

# تقسيم البيانات
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
# x_train تحتوي على صورة التدريب
x_train = np.array([item[0] for item in train_data])
# y_train تحتوي على تصنيف صورة التدريب
y_train = np.array([item[1] for item in train_data])
x_test = np.array([item[0] for item in test_data])
y_test = np.array([item[1] for item in test_data])

# تحديد النموذج
# تسمح بترتيب الطبقات التي تشكل النموذج بترتيب متسلسل
model = tf.keras.models.Sequential([
    # تطبيق تصفية على الصورة، حيث يتم تحديد 32 فلتر (filter) بحجم (3,3)، وتفعيلها باستخدام وظيفة التنشيط relu.
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                           input_shape=(224, 224, 3)),
    # تقوم بتقليل حجم الصورة عن طريق استخدام تقنية التجميع (pooling)
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    # تقوم بتحويل الصورة من صيغة مصفوفة ثنائية الأبعاد إلى مصفوفة واحدة ذات بُعد واحد.
    tf.keras.layers.Flatten(),
    #  هي طبقة كاملة الاتصال (fully connected) حيث يتم تحديد 128 عقدة
    tf.keras.layers.Dense(128, activation='relu'),
    # طبقة كاملة الاتصال تحتوي على 3 عقد بسبب أن هناك 3 فئات لتصنيف الصور
    tf.keras.layers.Dense(3, activation='softmax')
])

# تدريب النموذج
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=25, validation_data=(x_test, y_test))

# استخدام النموذج
# img = Image.open('sport_car_test.jpg')
img = Image.open('family_car_test.jpg')
# img = Image.open('small_car_test.jpg')
img = img.resize((224, 224))
img = np.array(img)
# تم تطبيع القيم في المصفوفة لتكون قيم بين 0 و 1.
img = img / 255.0
img = img.reshape((1,) + img.shape)
# سابقًا للتنبؤ بفئة الصورة التي تم تحليلها.
pred = model.predict(img)
#  الوظيفة argmax للعثور على الفئة الأكثر احتمالًا للصورة.
class_idx = tf.argmax(pred, axis=1)
class_label = ['sports car', 'family car', 'small car'][class_idx[0]]
print(class_label)
