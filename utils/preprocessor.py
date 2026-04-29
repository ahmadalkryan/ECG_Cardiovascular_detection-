# # utils/preprocessor.py
# """
# وحدة معالجة الصور المسبقة لتصنيف ECG
# تطبق نفس الخطوات المستخدمة في تدريب النموذج
# """

# import cv2
# import numpy as np
# from PIL import Image
# import io


# class ECGPreprocessor:
#     """
#     معالج صور ECG - يطبق نفس خطوات preprocessing المستخدمة في التدريب
    
#     الخطوات:
#     1. قص 18% من الأعلى و 6% من الأسفل
#     2. تحويل إلى Grayscale
#     3. CLAHE لتحسين التباين
#     4. Gaussian Blur لإزالة التشويش
#     5. تغيير الحجم إلى 224×224
#     6. تطبيع القيم إلى [0, 1]
#     7. إعادة إلى 3 قنوات
#     """
    
#     def __init__(self, target_size=(224, 224)):
#         """
#         تهيئة المعالج
        
#         Args:
#             target_size: الأبعاد النهائية للصورة (الافتراضي: 224×224)
#         """
#         self.target_size = target_size
#         self.top_crop = 0.18      # 18% من الأعلى
#         self.bottom_crop = 0.06   # 6% من الأسفل
    
#     def crop_top_bottom(self, image):
#         """
#         قص الأعلى والأسفل فقط (بدون قص جانبي)
        
#         Args:
#             image: صورة بتنسيق RGB numpy array
            
#         Returns:
#             الصورة بعد القص
#         """
#         h, w = image.shape[:2]
#         top = int(h * self.top_crop)
#         bottom = int(h * (1 - self.bottom_crop))
#         return image[top:bottom, :]
    
#     def preprocess(self, image):
#         """
#         تطبيق جميع خطوات المعالجة المسبقة
        
#         Args:
#             image: يمكن أن يكون:
#                 - مسار صورة (str)
#                 - مصفوفة numpy
#                 - PIL Image
#                 - bytes
                
#         Returns:
#             numpy array جاهز للإدخال إلى النموذج (1, 224, 224, 3)
#         """
#         # تحويل الإدخال إلى numpy array RGB
#         img = self._convert_to_numpy(image)
        
#         # 1. قص 18% من فوق و 6% من تحت
#         img = self.crop_top_bottom(img)
        
#         # 2. تحويل إلى Grayscale
#         gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
#         # 3. تطبيق CLAHE (تحسين التباين المحلي)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         enhanced = clahe.apply(gray)
        
#         # 4. Gaussian Blur (تنعيم لإزالة التشويش)
#         blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
#         # 5. إعادة إلى 3 قنوات (R, G, B متطابقة)
#         processed = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)
        
#         # 6. تغيير الحجم
#         resized = cv2.resize(processed, self.target_size)
        
#         # 7. تطبيع القيم إلى [0, 1]
#         normalized = resized.astype(np.float32) / 255.0
        
#         # 8. إضافة بُعد الدفعة (batch dimension)
#         batched = np.expand_dims(normalized, axis=0)
        
#         return batched
    
#     def _convert_to_numpy(self, image):
#         """
#         تحويل أنواع مختلفة من المدخلات إلى numpy array RGB
        
#         Args:
#             image: مسار، مصفوفة، PIL Image، أو bytes
            
#         Returns:
#             numpy array بتنسيق RGB
#         """
#         if isinstance(image, str):
#             # مسار صورة
#             img = cv2.imread(image)
#             if img is None:
#                 raise ValueError(f"Could not read image: {image}")
#             return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         elif isinstance(image, np.ndarray):
#             # مصفوفة numpy
#             if len(image.shape) == 2:
#                 # Grayscale → RGB
#                 return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
#             elif image.shape[2] == 3:
#                 if image.dtype == np.uint8:
#                     return image.copy()
#                 else:
#                     return (image * 255).astype(np.uint8)
#             else:
#                 raise ValueError(f"Unexpected image shape: {image.shape}")
        
#         elif isinstance(image, Image.Image):
#             # PIL Image
#             return np.array(image.convert('RGB'))
        
#         elif isinstance(image, bytes):
#             # Bytes
#             nparr = np.frombuffer(image, np.uint8)
#             img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#             return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         else:
#             raise ValueError(f"Unsupported image type: {type(image)}")
    
#     def get_preprocessing_stages(self, image):
#         """
#         الحصول على جميع مراحل المعالجة للعرض (لأغراض العرض التوضيحي)
        
#         Args:
#             image: الصورة الأصلية
            
#         Returns:
#             قاموس يحتوي على صور كل مرحلة
#         """
#         # تحويل الإدخال
#         img = self._convert_to_numpy(image)
        
#         stages = {}
        
#         # 1. الأصلية
#         stages['original'] = img.copy()
        
#         # 2. بعد القص
#         cropped = self.crop_top_bottom(img)
#         stages['cropped'] = cropped
        
#         # 3. Grayscale
#         gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
#         stages['grayscale'] = gray
        
#         # 4. CLAHE
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         enhanced = clahe.apply(gray)
#         stages['clahe'] = enhanced
        
#         # 5. Gaussian Blur
#         blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
#         stages['blurred'] = blurred
        
#         # 6. النهائية (بعد resize)
#         final = cv2.resize(blurred, self.target_size)
#         stages['final'] = final
        
#         return stages

# # في utils/preprocessor.py، أضف هذه الدالة
# def compare_original_processed(self, image):
#     """مقارنة الصورة الأصلية والمعالجة"""
    
#     original = self._convert_to_numpy(image)
#     processed = self.preprocess(image)[0]  # إزالة البعد الإضافي
    
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
#     axes[0].imshow(original)
#     axes[0].set_title(f'Original Image\nShape: {original.shape}')
#     axes[0].axis('off')
    
#     axes[1].imshow(processed)
#     axes[1].set_title(f'Processed Image\nShape: {processed.shape}\nRange: [{processed.min():.2f}, {processed.max():.2f}]')
#     axes[1].axis('off')
    
#     return fig

# # دالة مساعدة سريعة للاستخدام المباشر
# def preprocess_image(image, target_size=(224, 224)):
#     """
#     دالة سريعة لمعالجة الصورة (تغليف بسيط)
    
#     Args:
#         image: الصورة المدخلة
#         target_size: الأبعاد النهائية
        
#     Returns:
#         numpy array جاهز للنموذج
#     """
#     preprocessor = ECGPreprocessor(target_size)
#     return preprocessor.preprocess(image)


# utils/preprocessor.py - النسخة المصححة لـ Median Filter
import cv2
import numpy as np
from PIL import Image

class ECGPreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.top_crop = 0.18
        self.bottom_crop = 0.06
    
    def crop_top_bottom(self, image):
        h, w = image.shape[:2]
        top = int(h * self.top_crop)
        bottom = int(h * (1 - self.bottom_crop))
        return image[top:bottom, :]
    
    def preprocess(self, image):
        # تحويل إلى numpy
        if isinstance(image, Image.Image):
            img = np.array(image)
        elif isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()
        
        # 1. قص
        img = self.crop_top_bottom(img)
        
        # 2. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 3. ✅ MEDIAN FILTER (وليس CLAHE + Gaussian!)
        median = cv2.medianBlur(gray, 5)
        
        # 4. إعادة إلى 3 قنوات
        processed = cv2.cvtColor(median, cv2.COLOR_GRAY2RGB)
        
        # 5. تغيير الحجم
        resized = cv2.resize(processed, self.target_size)
        
        # 6. تطبيع
        normalized = resized.astype(np.float32) / 255.0
        
        # 7. إضافة بُعد الدفعة
        return np.expand_dims(normalized, axis=0)
    
    def _convert_to_numpy(self, image):
        if isinstance(image, str):
            img = cv2.imread(image)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            return np.array(image.convert('RGB'))
        return image
    
    def get_preprocessing_stages(self, image):
        img = self._convert_to_numpy(image)
        stages = {'original': img.copy()}
        
        cropped = self.crop_top_bottom(img)
        stages['cropped'] = cropped
        
        gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        stages['grayscale'] = gray
        
        median = cv2.medianBlur(gray, 5)
        stages['median'] = median
        
        final = cv2.resize(median, self.target_size)
        stages['final'] = final
        
        return stages