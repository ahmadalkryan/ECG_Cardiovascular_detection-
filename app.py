# # # app.py
# # """
# # تطبيق تصنيف صور ECG باستخدام Streamlit و ONNX Runtime
# # """

# # import streamlit as st
# # import numpy as np
# # import cv2
# # import plotly.graph_objects as go
# # import plotly.express as px
# # from PIL import Image
# # import time
# # from datetime import datetime
# # import pandas as pd
# # import os
# # import sys

# # # إضافة المسار للمكتبات المحلية
# # sys.path.append('.')

# # # استيراد المعالج
# # from utils.preprocessor import ECGPreprocessor

# # # =========================================================
# # # إعدادات الصفحة
# # # =========================================================
# # st.set_page_config(
# #     page_title="ECG Classification System",
# #     page_icon="🫀",
# #     layout="wide",
# #     initial_sidebar_state="expanded"
# # )

# # # =========================================================
# # # الثوابت والتعريفات
# # # =========================================================
# # CLASS_NAMES = ['Abnormal', 'MI', 'Normal', 'History_MI']

# # CLASS_DESCRIPTIONS = {
# #     'Abnormal': {
# #         'description': '🟡 **ضربات قلب غير طبيعية** - تشير إلى عدم انتظام في ضربات القلب قد يكون مرتبطًا بحالات مختلفة مثل عدم انتظام ضربات القلب (Arrhythmia).',
# #         'color': '#FFA500',
# #         'recommendation': '📋 يوصى باستشارة طبيب قلب لإجراء فحوصات إضافية',
# #         'urgency': 'متوسطة'
# #     },
# #     'MI': {
# #         'description': '🔴 **احتشاء عضلة القلب (Myocardial Infarction)** - يشير إلى وجود علامات تدل على نوبة قلبية حادة أو سابقة.',
# #         'color': '#FF0000',
# #         'recommendation': '⚠️ حالة طارئة - التوجه فوراً إلى أقرب مستشفى!',
# #         'urgency': 'طارئ'
# #     },
# #     'Normal': {
# #         'description': '🟢 **قلب طبيعي** - مخطط القلب ضمن الحدود الطبيعية. لا توجد علامات واضحة على وجود مشاكل قلبية.',
# #         'color': '#00FF00',
# #         'recommendation': '✅ الحفاظ على نمط حياة صحي ومتابعة دورية',
# #         'urgency': 'منخفضة'
# #     },
# #     'History_MI': {
# #         'description': '🟠 **تاريخ مرضي باحتشاء عضلة القلب** - يشير إلى وجود تغيرات في مخطط القلب تدل على تعرض المريض لنوبة قلبية سابقة.',
# #         'color': '#FF8C00',
# #         'recommendation': '📋 متابعة منتظمة مع طبيب القلب والالتزام بالعلاج',
# #         'urgency': 'عالية'
# #     }
# # }

# # # معلومات النموذج
# # MODEL_INFO = {
# #     'input_shape': (None, 224, 224, 3),
# #     'output_shape': (None, 4),
# #     'num_layers': 9,
# #     'total_params': 7_337_412,
# #     'test_accuracy': 94.29,
# #     'best_val_accuracy': 96.40
# # }

# # # =========================================================
# # # تحميل النموذج
# # # =========================================================
# # @st.cache_resource
# # def load_onnx_model(model_path):
# #     """
# #     تحميل نموذج ONNX باستخدام ONNX Runtime
    
# #     Args:
# #         model_path: مسار ملف النموذج
        
# #     Returns:
# #         كائن InferenceSession
# #     """
# #     try:
# #         import onnxruntime as ort
        
# #         # التحقق من وجود الملف
# #         if not os.path.exists(model_path):
# #             st.error(f"❌ النموذج غير موجود: {model_path}")
# #             st.info("تأكد من وجود ملف ecg_median_model.onnx في مجلد models/")
# #             return None
        
# #         # إنشاء جلسة ONNX
# #         sess_options = ort.SessionOptions()
# #         sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
# #         # تحميل النموذج
# #         session = ort.InferenceSession(model_path, sess_options)
        
# #         # طباعة معلومات النموذج
# #         st.success(f"✅ تم تحميل النموذج بنجاح!")
        
# #         return session
        
# #     except Exception as e:
# #         st.error(f"❌ خطأ في تحميل النموذج: {e}")
# #         return None


# # def predict_with_onnx(session, input_data):
# #     """
# #     تنفيذ التنبؤ باستخدام نموذج ONNX
    
# #     Args:
# #         session: جلسة ONNX
# #         input_data: البيانات المدخلة (1, 224, 224, 3)
        
# #     Returns:
# #         مصفوفة الاحتمالات
# #     """
# #     try:
# #         # الحصول على اسم الإدخال
# #         input_name = session.get_inputs()[0].name
        
# #         # تنفيذ التنبؤ
# #         outputs = session.run(None, {input_name: input_data.astype(np.float32)})
        
# #         # النتيجة هي أول مخرج
# #         predictions = outputs[0][0]
        
# #         return predictions
        
# #     except Exception as e:
# #         st.error(f"❌ خطأ في التنبؤ: {e}")
# #         return None


# # # =========================================================
# # # دوال عرض النتائج
# # # =========================================================
# # def display_results(result):
# #     """
# #     عرض نتائج التنبؤ بطريقة جميلة
    
# #     Args:
# #         result: قاموس يحتوي على نتائج التنبؤ
# #     """
# #     predicted_class = result['class']
# #     confidence = result['confidence']
# #     probabilities = result['probabilities']
    
# #     # معلومات الفئة
# #     class_info = CLASS_DESCRIPTIONS[predicted_class]
    
# #     # تنسيق اللون للخلفية
# #     bg_color = f"rgba({int(class_info['color'][1:3], 16)}, {int(class_info['color'][3:5], 16)}, {int(class_info['color'][5:7], 16)}, 0.1)"
    
# #     # عرض النتيجة الرئيسية
# #     st.markdown(f"""
# #     <div style="
# #         background: {bg_color};
# #         border-left: 5px solid {class_info['color']};
# #         border-radius: 10px;
# #         padding: 20px;
# #         margin: 10px 0;
# #     ">
# #         <div style="display: flex; justify-content: space-between; align-items: center;">
# #             <div>
# #                 <h2 style="color: {class_info['color']}; margin: 0;">
# #                     🩺 {predicted_class}
# #                 </h2>
# #                 <p style="margin: 5px 0; color: #666;">
# #                     مستوى الخطورة: <strong style="color: {class_info['color']};">{class_info['urgency']}</strong>
# #                 </p>
# #             </div>
# #             <div style="text-align: center;">
# #                 <div style="font-size: 48px; font-weight: bold; color: {class_info['color']};">
# #                     {confidence:.1f}%
# #                 </div>
# #                 <div style="font-size: 14px; color: #666;">نسبة الثقة</div>
# #             </div>
# #         </div>
# #         <p style="margin-top: 15px;">
# #             {class_info['description']}
# #         </p>
# #         <div style="
# #             background: rgba(0,0,0,0.05);
# #             padding: 10px;
# #             border-radius: 8px;
# #             margin-top: 10px;
# #         ">
# #             <strong>💡 التوصية:</strong> {class_info['recommendation']}
# #         </div>
# #     </div>
# #     """, unsafe_allow_html=True)
    
# #     # رسم بياني للاحتمالات
# #     fig = go.Figure(data=[
# #         go.Bar(
# #             x=list(probabilities.keys()),
# #             y=list(probabilities.values()),
# #             marker_color=[CLASS_DESCRIPTIONS[c]['color'] for c in probabilities.keys()],
# #             text=[f"{v:.1f}%" for v in probabilities.values()],
# #             textposition='outside',
# #             textfont=dict(size=12)
# #         )
# #     ])
    
# #     fig.update_layout(
# #         title="احتمالات التصنيف",
# #         xaxis_title="الفئة",
# #         yaxis_title="النسبة المئوية (%)",
# #         yaxis_range=[0, 100],
# #         height=400,
# #         showlegend=False,
# #         plot_bgcolor='rgba(0,0,0,0)',
# #         paper_bgcolor='rgba(0,0,0,0)'
# #     )
    
# #     fig.update_traces(
# #         marker=dict(line=dict(width=1, color='DarkSlateGrey'))
# #     )
    
# #     st.plotly_chart(fig, use_container_width=True)


# # def display_preprocessing_stages(preprocessor, image):
# #     """
# #     عرض مراحل المعالجة المسبقة
    
# #     Args:
# #         preprocessor: كائن المعالج
# #         image: الصورة الأصلية
# #     """
# #     st.markdown("### 🔬 مراحل معالجة الصورة")
    
# #     with st.spinner("جاري تحليل مراحل المعالجة..."):
# #         stages = preprocessor.get_preprocessing_stages(image)
    
# #     # عرض المراحل في أعمدة
# #     cols = st.columns(len(stages))
    
# #     stage_names = {
# #         'original': '📷 الصورة الأصلية',
# #         'cropped': '✂️ بعد القص (18%↑ + 6%↓)',
# #         'grayscale': '⚫ Grayscale',
# #         'clahe': '🎨 CLAHE (تحسين التباين)',
# #         'blurred': '🌫️ Gaussian Blur',
# #         'final': '✅ الصورة النهائية (224×224)'
# #     }
    
# #     for idx, (name, stage_img) in enumerate(stages.items()):
# #         with cols[idx]:
# #             if len(stage_img.shape) == 3:
# #                 st.image(stage_img, use_container_width=True)
# #             else:
# #                 st.image(stage_img, use_container_width=True, clamp=True)
# #             st.caption(stage_names.get(name, name))
    
# #     # معلومات إضافية
# #     with st.expander("📋 تفاصيل خطوات المعالجة"):
# #         st.markdown("""
# #         | الخطوة | الوصف | الهدف |
# #         |:-------|:------|:------|
# #         | **القص** | إزالة 18% من الأعلى و 6% من الأسفل | إزالة النصوص والعناوين |
# #         | **Grayscale** | تحويل الصورة إلى تدرج رمادي | تقليل الأبعاد (3→1) |
# #         | **CLAHE** | تحسين التباين المحلي | جعل الإشارة أوضح |
# #         | **Gaussian Blur** | تنعيم الصورة | إزالة التشويش |
# #         | **Resize** | تغيير الحجم إلى 224×224 | توحيد الأبعاد للنموذج |
# #         | **Normalize** | تحويل القيم إلى [0,1] | تحسين استقرار النموذج |
# #         """)


# # def display_model_info():
# #     """عرض معلومات النموذج التقنية"""
# #     st.markdown("### 🏗️ معلومات النموذج التقنية")
    
# #     col1, col2 = st.columns(2)
    
# #     with col1:
# #         st.markdown(f"""
# #         | الخاصية | القيمة |
# #         |:---------|:-------|
# #         | **النموذج** | DenseNet121 (ONNX) |
# #         | **حجم الإدخال** | 224 × 224 × 3 |
# #         | **حجم الإخراج** | 4 فئات |
# #         | **عدد الطبقات** | {MODEL_INFO['num_layers']} |
# #         | **المعاملات** | {MODEL_INFO['total_params']:,} |
# #         """)
    
# #     with col2:
# #         st.markdown(f"""
# #         | المقياس | القيمة |
# #         |:---------|:-------|
# #         | **دقة الاختبار** | {MODEL_INFO['test_accuracy']:.2f}% |
# #         | **أفضل دقة تحقق** | {MODEL_INFO['best_val_accuracy']:.2f}% |
# #         | **دقة MI** | 100% |
# #         | **دقة Abnormal** | 100% |
# #         """)
    
# #     st.info("""
# #     💡 **ملاحظة:** النموذج تم تدريبه على 928 صورة ECG وتم تحويله إلى صيغة ONNX 
# #     لتحسين الأداء والتوافق مع تطبيقات .NET.
# #     """)


# # # =========================================================
# # # الواجهة الرئيسية
# # # =========================================================
# # def main():
# #     """الدالة الرئيسية للتطبيق"""
    
# #     # الشريط الجانبي
# #     with st.sidebar:
# #         st.image("https://cdn-icons-png.flaticon.com/512/2972/2972185.png", width=80)
# #         st.title("🫀 ECG Classification")
# #         st.markdown("---")
        
# #         st.markdown("""
# #         ### 📌 عن النظام
# #         نظام ذكي لتحليل وتصنيف مخططات القلب (ECG) باستخدام **Deep Learning**.
        
# #         **الفئات المدعومة:**
# #         - 🟡 Abnormal (ضربات غير طبيعية)
# #         - 🔴 MI (احتشاء عضلة القلب)
# #         - 🟢 Normal (طبيعي)
# #         - 🟠 History_MI (تاريخ مرضي)
        
# #         ---
        
# #         ### 🧠 التقنيات المستخدمة
# #         - 🖥️ **Streamlit** - واجهة المستخدم
# #         - 🚀 **ONNX Runtime** - تشغيل النموذج
# #         - 🎨 **OpenCV** - معالجة الصور
# #         - 📊 **Plotly** - الرسوم البيانية
        
# #         ---
        
# #         ### 📊 أداء النموذج
# #         - **الدقة:** 94.29%
# #         - **MI (النوبة القلبية):** 100%
# #         - **زمن التنبؤ:** ~50ms
# #         """)
        
# #         st.markdown("---")
# #         st.caption("⚠️ هذا النظام للمساعدة في التشخيص وليس بديلاً عن الاستشارة الطبية")
    
# #     # المحتوى الرئيسي
# #     st.title("🫀 **تحليل وتصنيف مخططات القلب (ECG)**")
# #     st.markdown("قم برفع صورة مخطط القلب للحصول على تحليل فوري وتصنيف ذكي.")
# #     st.markdown("---")
    
# #     # تحميل النموذج
# #     model_path = os.path.join("models", "ecg_median_model.onnx")
# #     session = load_onnx_model(model_path)
    
# #     if session is None:
# #         st.stop()
    
# #     # تهيئة المعالج
# #     preprocessor = ECGPreprocessor()
    
# #     # تبويبات
# #     tab1, tab2, tab3 = st.tabs(["📤 رفع الصورة وتحليلها", "📊 معلومات النموذج", "ℹ️ تعليمات الاستخدام"])
    
# #     with tab1:
# #         # رفع الصورة
# #         uploaded_file = st.file_uploader(
# #             "اختر صورة مخطط قلب (ECG)",
# #             type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
# #             help="الصورة يجب أن تكون واضحة وتظهر إشارة ECG بشكل جيد"
# #         )
        
# #         # خيارات إضافية
# #         col1, col2 = st.columns(2)
# #         with col1:
# #             show_preprocessing = st.checkbox("🔬 عرض مراحل المعالجة", value=False)
# #         with col2:
# #             show_progress = st.checkbox("📊 عرض شريط التقدم", value=True)
        
# #         if uploaded_file is not None:
# #             # قراءة الصورة
# #             image = Image.open(uploaded_file)
            
# #             # عرض الصورة
# #             col_img1, col_img2 = st.columns([1, 1])
# #             with col_img1:
# #                 st.image(image, use_container_width=True, caption="📷 الصورة المرفوعة")
            
# #             with col_img2:
# #                 st.info(f"""
# #                 **📋 معلومات الصورة:**
# #                 - الحجم: {image.size[0]} × {image.size[1]} بكسل
# #                 - الصيغة: {image.format}
# #                 - الوضع: {image.mode}
# #                 """)
            
# #             # زر التنبؤ
# #             if st.button("🚀 **بدء التحليل**", type="primary", use_container_width=True):
                
# #                 if show_progress:
# #                     progress_bar = st.progress(0)
# #                     status_text = st.empty()
                    
# #                     status_text.text("📂 جاري قراءة الصورة...")
# #                     progress_bar.progress(10)
# #                     time.sleep(0.2)
                    
# #                     status_text.text("✂️ جاري قص الصورة...")
# #                     progress_bar.progress(25)
# #                     time.sleep(0.2)
                    
# #                     status_text.text("🎨 جاري معالجة الصورة و تحسين التباين...")
# #                     progress_bar.progress(50)
# #                     time.sleep(0.2)
                    
# #                     status_text.text("🧠 جاري تحليل الإشارة بواسطة النموذج...")
# #                     progress_bar.progress(75)
# #                     time.sleep(0.2)
                    
# #                     status_text.text("✅ جاري إنشاء التقرير...")
# #                     progress_bar.progress(90)
# #                     time.sleep(0.2)
                
# #                 # تنفيذ التنبؤ
# #                 try:
# #                     start_time = time.time()
                    
# #                     # المعالجة المسبقة
# #                     input_data = preprocessor.preprocess(image)
                    
# #                     # التنبؤ
# #                     predictions = predict_with_onnx(session, input_data)
                    
# #                     inference_time = (time.time() - start_time) * 1000
                    
# #                     if predictions is not None:
# #                         # تفسير النتائج
# #                         predicted_class = CLASS_NAMES[np.argmax(predictions)]
# #                         confidence = np.max(predictions) * 100
# #                         probabilities = {name: prob * 100 for name, prob in zip(CLASS_NAMES, predictions)}
                        
# #                         result = {
# #                             'class': predicted_class,
# #                             'confidence': confidence,
# #                             'probabilities': probabilities,
# #                             'success': True
# #                         }
                        
# #                         if show_progress:
# #                             progress_bar.progress(100)
# #                             status_text.text("✅ اكتمل التحليل!")
# #                             time.sleep(0.5)
# #                             progress_bar.empty()
# #                             status_text.empty()
                        
# #                         # عرض النتائج
# #                         st.markdown("---")
# #                         st.subheader("📊 **نتائج التحليل**")
# #                         st.caption(f"⏱️ وقت التحليل: {inference_time:.0f} ملي ثانية")
                        
# #                         display_results(result)
                        
# #                         if show_preprocessing:
# #                             st.markdown("---")
# #                             display_preprocessing_stages(preprocessor, image)
                        
# #                         # نصائح إضافية
# #                         if predicted_class == "MI":
# #                             st.error("""
# #                             🚨 **تنبيه هام:** النتيجة تشير إلى احتمال وجود احتشاء عضلة القلب.
# #                             يرجى التوجه فوراً إلى أقرب مستشفى أو الاتصال بالإسعاف!
# #                             """)
# #                         elif predicted_class == "Abnormal":
# #                             st.warning("""
# #                             ⚠️ **تنبيه:** النتيجة تشير إلى وجود ضربات قلب غير طبيعية.
# #                             يرجى استشارة طبيب قلب في أقرب وقت.
# #                             """)
# #                         elif predicted_class == "History_MI":
# #                             st.info("""
# #                             ℹ️ **ملاحظة:** النتيجة تشير إلى وجود تاريخ مرضي باحتشاء عضلة القلب.
# #                             ينصح بالمتابعة المنتظمة مع طبيب القلب.
# #                             """)
# #                         else:
# #                             st.success("""
# #                             ✅ **نتيجة مطمئنة:** مخطط القلب ضمن الحدود الطبيعية.
# #                             حافظ على نمط حياة صحي.
# #                             """)
                            
# #                 except Exception as e:
# #                     st.error(f"❌ حدث خطأ أثناء التحليل: {e}")
    
# #     with tab2:
# #         display_model_info()
    
# #     with tab3:
# #         st.markdown("""
# #         ### 📖 تعليمات الاستخدام
        
# #         #### 1. رفع الصورة
# #         - اضغط على زر "اختر صورة مخطط قلب"
# #         - اختر صورة ECG من جهازك
# #         - الصيغ المدعومة: JPG, PNG, BMP, TIFF
        
# #         #### 2. التحليل
# #         - اضغط على زر "بدء التحليل"
# #         - انتظر بضع ثوانٍ للحصول على النتائج
        
# #         #### 3. تفسير النتائج
# #         - **Normal**: مخطط طبيعي - لا توجد مشاكل
# #         - **Abnormal**: ضربات غير طبيعية - استشارة طبيب
# #         - **MI**: نوبة قلبية - حالة طارئة!
# #         - **History_MI**: تاريخ مرضي - متابعة مستمرة
        
# #         #### ⚠️ تنبيهات مهمة
# #         - هذا النظام هو أداة مساعدة وليس بديلاً عن التشخيص الطبي
# #         - دقة النموذج 94.29%، قد تكون هناك أخطاء
# #         - في حالة الشك، استشر طبيباً مختصاً
        
# #         #### 📞 الدعم الفني
# #         في حالة وجود مشاكل تقنية، يرجى مراجعة:
# #         - تأكد من وجود ملف النموذج في مجلد models/
# #         - تأكد من تثبيت جميع المكتبات المطلوبة
# #         """)
    
# #     # Footer
# #     st.markdown("---")
# #     st.markdown(
# #         "<div style='text-align: center; color: #666;'>"
# #         "🫀 نظام تصنيف ECG باستخدام الذكاء الاصطناعي | للإستخدام المساعد فقط"
# #         "</div>",
# #         unsafe_allow_html=True
# #     )


# # if __name__ == "__main__":
# #     main()


# # app.py - نسخة مبسطة ومصححة
# import streamlit as st
# import numpy as np
# import cv2
# import plotly.graph_objects as go
# from PIL import Image
# import time
# import os
# import sys

# sys.path.append('.')
# from utils.preprocessor import ECGPreprocessor

# # إعدادات الصفحة
# st.set_page_config(
#     page_title="ECG Classification System",
#     page_icon="🫀",
#     layout="wide"
# )

# # تعريفات الفئات
# CLASS_NAMES = ['Abnormal', 'MI', 'Normal', 'History_MI']

# CLASS_COLORS = {
#     'Abnormal': '#FFA500',
#     'MI': '#FF0000',
#     'Normal': '#00FF00',
#     'History_MI': '#FF8C00'
# }

# # تحميل النموذج
# @st.cache_resource
# def load_onnx_model(model_path):
#     import onnxruntime as ort
#     if not os.path.exists(model_path):
#         st.error(f"❌ النموذج غير موجود: {model_path}")
#         return None
#     return ort.InferenceSession(model_path)

# # دالة التنبؤ
# def predict(session, input_data):
#     input_name = session.get_inputs()[0].name
#     outputs = session.run(None, {input_name: input_data.astype(np.float32)})
#     return outputs[0][0]

# # عرض النتائج
# def display_results(predictions):
#     predicted_idx = np.argmax(predictions)
#     predicted_class = CLASS_NAMES[predicted_idx]
#     confidence = predictions[predicted_idx] * 100
    
#     # عرض النتيجة
#     st.markdown(f"""
#     <div style="
#         border-left: 5px solid {CLASS_COLORS[predicted_class]};
#         padding: 20px;
#         border-radius: 10px;
#         margin: 10px 0;
#         background: rgba(0,0,0,0.05);
#     ">
#         <h2 style="color: {CLASS_COLORS[predicted_class]}; margin: 0;">
#             🩺 {predicted_class}
#         </h2>
#         <h1>{confidence:.1f}% <span style="font-size: 16px;">ثقة</span></h1>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # رسم بياني
#     fig = go.Figure(data=[
#         go.Bar(
#             x=CLASS_NAMES,
#             y=predictions * 100,
#             marker_color=[CLASS_COLORS[c] for c in CLASS_NAMES],
#             text=[f"{p*100:.1f}%" for p in predictions],
#             textposition='outside'
#         )
#     ])
#     fig.update_layout(
#         title="احتمالات التصنيف",
#         yaxis_range=[0, 100],
#         height=400
#     )
#     st.plotly_chart(fig, use_container_width=True)


# # أضف هذه الدالة في app.py
# def debug_prediction(input_data, predictions):
#     """عرض معلومات التصحيح للتنبؤ"""
    
#     st.markdown("### 🔍 معلومات التصحيح")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("**📊 معلومات الإدخال:**")
#         st.write(f"- الشكل (Shape): {input_data.shape}")
#         st.write(f"- نوع البيانات (dtype): {input_data.dtype}")
#         st.write(f"- المدى (min/max): {input_data.min():.3f} / {input_data.max():.3f}")
#         st.write(f"- المتوسط (mean): {input_data.mean():.3f}")
#         st.write(f"- الانحراف المعياري (std): {input_data.std():.3f}")
    
#     with col2:
#         st.markdown("**🎯 مخرجات النموذج:**")
#         for name, prob in zip(CLASS_NAMES, predictions):
#             st.write(f"- {name}: {prob*100:.2f}%")
#         st.write(f"- الفئة المتوقعة: {CLASS_NAMES[np.argmax(predictions)]}")
#         st.write(f"- أعلى ثقة: {np.max(predictions)*100:.2f}%")
#         st.write(f"- أدنى ثقة: {np.min(predictions)*100:.2f}%")
    
#     # رسم توزيع الاحتمالات
#     fig, ax = plt.subplots(figsize=(8, 4))
#     bars = ax.bar(CLASS_NAMES, predictions * 100, color=['#FFA500', '#FF0000', '#00FF00', '#FF8C00'])
#     ax.set_ylabel('النسبة (%)')
#     ax.set_title('توزيع احتمالات النموذج')
#     ax.set_ylim(0, 100)
    
#     # إضافة القيم على الأعمدة
#     for bar, prob in zip(bars, predictions):
#         ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
#                 f'{prob*100:.1f}%', ha='center', va='bottom')
    
#     st.pyplot(fig)
# # =========================================================
# # الواجهة الرئيسية
# # =========================================================
# def main():
#     st.title("🫀 ECG Classification System")
#     st.markdown("---")
    
#     # تحميل النموذج
#     session = load_onnx_model("models/ecg_median_model.onnx")
#     if session is None:
#         st.stop()
    
#     preprocessor = ECGPreprocessor()
    
#     # رفع الصورة
#     uploaded_file = st.file_uploader(
#         "اختر صورة ECG",
#         type=['jpg', 'jpeg', 'png', 'bmp']
#     )
    
#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
        
#         # ✅ استخدام use_column_width بدلاً من use_container_width
#         st.image(image, use_column_width=True, caption="الصورة المرفوعة")
        
#         # زر التحليل
#         if st.button("بدء التحليل", type="primary"):
#             with st.spinner("جاري التحليل..."):
#                 # المعالجة والتنبؤ
#                 input_data = preprocessor.preprocess(image)
#                 predictions = predict(session, input_data)
                
#                 # عرض النتائج
#                 st.markdown("---")
#                 st.subheader("نتائج التحليل")
#                 display_results(predictions)

# if __name__ == "__main__":
#     main()






# app.py - النسخة المصححة بالكامل
import streamlit as st
import numpy as np
import cv2
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import sys

sys.path.append('.')
from utils.preprocessor import ECGPreprocessor

# إعدادات الصفحة
st.set_page_config(
    page_title="ECG Classification System",
    page_icon="🫀",
    layout="wide"
)

# تعريفات الفئات
CLASS_NAMES = ['Abnormal', 'MI', 'Normal', 'History_MI']

CLASS_COLORS = {
    'Abnormal': '#FFA500',
    'MI': '#FF0000',
    'Normal': '#00FF00',
    'History_MI': '#FF8C00'
}

# =========================================================
# تحميل النموذج مع معلومات تفصيلية
# =========================================================
@st.cache_resource
def load_onnx_model(model_path):
    """تحميل نموذج ONNX مع معلومات تفصيلية"""
    import onnxruntime as ort
    
    if not os.path.exists(model_path):
        st.error(f"❌ النموذج غير موجود: {model_path}")
        st.info("تأكد من وجود ملف النموذج في مجلد models/")
        return None
    
    # تحميل النموذج
    session = ort.InferenceSession(model_path)
    
    # عرض معلومات النموذج
    with st.expander("📊 معلومات النموذج"):
        st.write(f"**Input name:** {session.get_inputs()[0].name}")
        st.write(f"**Input shape:** {session.get_inputs()[0].shape}")
        st.write(f"**Output name:** {session.get_outputs()[0].name}")
        st.write(f"**Output shape:** {session.get_outputs()[0].shape}")
    
    return session

# =========================================================
# دالة التنبؤ مع Debugging
# =========================================================
def predict_with_debug(session, input_data):
    """تنبؤ مع معلومات تصحيح مفصلة"""
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data.astype(np.float32)})
    predictions = outputs[0][0]
    
    # معلومات التصحيح
    debug_info = {
        'input_shape': input_data.shape,
        'input_min': input_data.min(),
        'input_max': input_data.max(),
        'input_mean': input_data.mean(),
        'input_std': input_data.std(),
        'predictions': predictions,
        'predicted_class': CLASS_NAMES[np.argmax(predictions)],
        'confidence': np.max(predictions) * 100,
        'all_probs': {name: prob * 100 for name, prob in zip(CLASS_NAMES, predictions)}
    }
    
    return predictions, debug_info

# =========================================================
# عرض معلومات التصحيح
# =========================================================
def display_debug_info(debug_info):
    """عرض معلومات التصحيح"""
    with st.expander("🔍 معلومات التصحيح (Debug Information)"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 بيانات الإدخال:**")
            st.write(f"- الشكل: {debug_info['input_shape']}")
            st.write(f"- المدى: [{debug_info['input_min']:.3f}, {debug_info['input_max']:.3f}]")
            st.write(f"- المتوسط: {debug_info['input_mean']:.3f}")
            st.write(f"- الانحراف المعياري: {debug_info['input_std']:.3f}")
        
        with col2:
            st.markdown("**🎯 مخرجات النموذج:**")
            for name, prob in debug_info['all_probs'].items():
                st.write(f"- {name}: {prob:.2f}%")
            st.write(f"- **الفئة المتوقعة:** {debug_info['predicted_class']}")
            st.write(f"- **نسبة الثقة:** {debug_info['confidence']:.2f}%")
        
        # تحليل صحة النموذج
        st.markdown("**🔬 تحليل صحة النموذج:**")
        predictions = debug_info['predictions']
        max_prob = np.max(predictions)
        second_max = np.sort(predictions)[-2]
        
        if max_prob - second_max < 0.1:
            st.warning("⚠️ النموذج غير واثق من قراره (الفجوة بين أعلى احتمال والثاني صغيرة جداً)")
        elif max_prob < 0.5:
            st.warning("⚠️ أعلى احتمال أقل من 50%")
        else:
            st.success("✅ النموذج واثق من قراره")

# =========================================================
# عرض النتائج
# =========================================================
def display_results(predictions):
    """عرض نتائج التنبؤ"""
    predicted_idx = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = predictions[predicted_idx] * 100
    
    # عرض النتيجة الرئيسية
    st.markdown(f"""
    <div style="
        border-left: 5px solid {CLASS_COLORS[predicted_class]};
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        background: rgba(0,0,0,0.05);
    ">
        <h2 style="color: {CLASS_COLORS[predicted_class]}; margin: 0;">
            🩺 {predicted_class}
        </h2>
        <h1>{confidence:.1f}% <span style="font-size: 16px;">ثقة</span></h1>
    </div>
    """, unsafe_allow_html=True)
    
    # رسم بياني
    fig = go.Figure(data=[
        go.Bar(
            x=CLASS_NAMES,
            y=predictions * 100,
            marker_color=[CLASS_COLORS[c] for c in CLASS_NAMES],
            text=[f"{p*100:.1f}%" for p in predictions],
            textposition='outside'
        )
    ])
    fig.update_layout(
        title="احتمالات التصنيف",
        yaxis_range=[0, 100],
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# عرض مراحل المعالجة
# =========================================================
def display_preprocessing_stages(preprocessor, image):
    """عرض مراحل المعالجة"""
    st.markdown("### 🔬 مراحل معالجة الصورة")
    
    with st.spinner("جاري تحليل مراحل المعالجة..."):
        stages = preprocessor.get_preprocessing_stages(image)
    
    cols = st.columns(len(stages))
    
    stage_names = {
        'original': '📷 الصورة الأصلية',
        'cropped': '✂️ بعد القص',
        'grayscale': '⚫ Grayscale',
        'clahe': '🎨 CLAHE',
        'blurred': '🌫️ Gaussian Blur',
        'final': '✅ الصورة النهائية'
    }
    
    for idx, (name, stage_img) in enumerate(stages.items()):
        with cols[idx]:
            if len(stage_img.shape) == 3:
                st.image(stage_img, use_column_width=True)
            else:
                st.image(stage_img, use_column_width=True, clamp=True)
            st.caption(stage_names.get(name, name))

# =========================================================
# عرض معلومات النموذج
# =========================================================
def display_model_info(session):
    """عرض معلومات النموذج"""
    st.markdown("### 🏗️ معلومات النموذج")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📥 المدخلات:**")
        st.write(f"- الاسم: {session.get_inputs()[0].name}")
        st.write(f"- الشكل: {session.get_inputs()[0].shape}")
        st.write(f"- النوع: {session.get_inputs()[0].type}")
    
    with col2:
        st.markdown("**📤 المخرجات:**")
        st.write(f"- الاسم: {session.get_outputs()[0].name}")
        st.write(f"- الشكل: {session.get_outputs()[0].shape}")
        st.write(f"- الفئات: {CLASS_NAMES}")

# =========================================================
# دالة اختبار النموذج ببيانات عشوائية
# =========================================================
def test_model_random(session):
    """اختبار النموذج ببيانات عشوائية للتحقق من عمله"""
    st.markdown("### 🧪 اختبار النموذج ببيانات عشوائية")
    
    # إنشاء بيانات عشوائية
    random_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    
    # تنبؤ
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: random_input})[0][0]
    
    st.write("**نتائج الاختبار:**")
    st.write(f"- أعلى احتمال: {np.max(output)*100:.2f}%")
    st.write(f"- الفئة المتوقعة: {CLASS_NAMES[np.argmax(output)]}")
    
    # التحقق من تنوع المخرجات
    if len(np.unique(output)) < 2:
        st.error("❌ **تحذير:** النموذج يعطي نفس المخرجات تقريباً لكل المدخلات!")
        st.error("   هذا يشير إلى أن النموذج قد يكون تالفاً أو محولاً بشكل خاطئ.")
        return False
    else:
        st.success("✅ النموذج يستجيب بشكل مختلف للمدخلات المختلفة - جيد!")
        return True

# =========================================================
# الواجهة الرئيسية
# =========================================================
def main():
    st.title("🫀 ECG Classification System")
    st.markdown("---")
    
    # الشريط الجانبي
    with st.sidebar:
        st.markdown("### 🫀 ECG Classification")
        st.markdown("---")
        st.markdown("""
        **الفئات:**
        - 🟡 Abnormal (ضربات غير طبيعية)
        - 🔴 MI (احتشاء عضلة القلب)
        - 🟢 Normal (طبيعي)
        - 🟠 History_MI (تاريخ مرضي)
        
        **المعالجة:**
        - قص 18% فوق + 6% تحت
        - Median Filter (kernel=5)
        - تغيير الحجم 224×224
        """)
    
    # تحميل النموذج
    model_path = os.path.join("models", "ecg_median_model.onnx")
    
    # إذا لم يكن موجوداً، جرب مساراً آخر
    if not os.path.exists(model_path):
        model_path = "ecg_median_model.onnx"
    
    session = load_onnx_model(model_path)
    
    if session is None:
        st.error("❌ لم يتم العثور على النموذج!")
        st.info("""
        **الحلول:**
        1. تأكد من وجود ملف النموذج في مجلد `models/`
        2. أو ضع الملف في نفس مجلد `app.py`
        3. تأكد من اسم الملف: `ecg_median_model.onnx`
        """)
        return
    
    # اختبار النموذج
    with st.expander("🧪 اختبار صحة النموذج"):
        test_model_random(session)
        display_model_info(session)
    
    # تهيئة المعالج
    preprocessor = ECGPreprocessor()
    
    # رفع الصورة
    uploaded_file = st.file_uploader(
        "📤 اختر صورة ECG للتحليل",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff']
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # عرض الصورة
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, use_column_width=True, caption="📷 الصورة المرفوعة")
        
        with col2:
            st.info(f"""
            **📋 معلومات الصورة:**
            - الحجم: {image.size[0]} × {image.size[1]} بكسل
            - الصيغة: {image.format}
            """)
        
        # خيارات
        show_preprocessing = st.checkbox("🔬 عرض مراحل المعالجة", value=True)
        show_debug = st.checkbox("🔍 عرض معلومات التصحيح", value=True)
        
        # زر التحليل
        if st.button("🚀 بدء التحليل", type="primary"):
            with st.spinner("جاري التحليل..."):
                try:
                    start_time = time.time()
                    
                    # المعالجة المسبقة
                    input_data = preprocessor.preprocess(image)
                    
                    # التنبؤ
                    predictions, debug_info = predict_with_debug(session, input_data)
                    
                    inference_time = (time.time() - start_time) * 1000
                    
                    # عرض النتائج
                    st.markdown("---")
                    st.subheader(f"📊 نتائج التحليل (⏱️ {inference_time:.0f}ms)")
                    
                    display_results(predictions)
                    
                    if show_preprocessing:
                        st.markdown("---")
                        display_preprocessing_stages(preprocessor, image)
                    
                    if show_debug:
                        display_debug_info(debug_info)
                    
                    # نصائح حسب النتيجة
                    predicted_class = CLASS_NAMES[np.argmax(predictions)]
                    if predicted_class == "MI":
                        st.error("""
                        🚨 **تنبيه هام:** النتيجة تشير إلى احتمال وجود احتشاء عضلة القلب.
                        هذه حالة طارئة تستدعي التدخل الطبي الفوري!
                        """)
                    elif predicted_class == "Abnormal":
                        st.warning("""
                        ⚠️ **تنبيه:** تم اكتشاف ضربات قلب غير طبيعية.
                        يرجى استشارة طبيب قلب في أقرب وقت.
                        """)
                    elif predicted_class == "History_MI":
                        st.info("""
                        ℹ️ **ملاحظة:** توجد علامات تشير إلى تاريخ مرضي باحتشاء عضلة القلب.
                        ينصح بالمتابعة المنتظمة مع طبيب القلب.
                        """)
                    else:
                        st.success("""
                        ✅ **نتيجة مطمئنة:** مخطط القلب ضمن الحدود الطبيعية.
                        حافظ على نمط حياة صحي.
                        """)
                        
                except Exception as e:
                    st.error(f"❌ حدث خطأ أثناء التحليل: {str(e)}")
                    st.exception(e)

if __name__ == "__main__":
    main()