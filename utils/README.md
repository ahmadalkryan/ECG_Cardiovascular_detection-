# 🫀 ECG Classification System

<div align="center">

![ECG Banner](https://img.shields.io/badge/ECG-Classification-blue?style=for-the-badge&logo=heart)
![Python](https://img.shields.io/badge/Python-3.9+-green?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red?style=flat-square&logo=streamlit)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-yellow?style=flat-square&logo=onnx)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

**نظام ذكي لتحليل وتصنيف مخططات القلب (ECG) باستخدام Deep Learning**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]()

</div>

---

## 📋 **نبذة عن المشروع**

هذا المشروع يقدم تطبيقاً تفاعلياً لتحليل وتصنيف صور مخططات القلب (ECG) باستخدام نموذج **Deep Learning** من نوع **DenseNet121** مع **Fine-tuning**. النظام قادر على تصنيف مخططات القلب إلى 4 فئات:

|     🟡 Abnormal      |             🔴 MI              | 🟢 Normal |   🟠 History_MI    |
| :------------------: | :----------------------------: | :-------: | :----------------: |
| ضربات قلب غير طبيعية | احتشاء عضلة القلب (نوبة قلبية) | قلب طبيعي | تاريخ مرضي باحتشاء |

---

## ✨ **المميزات**

- 🖥️ **واجهة مستخدم سهلة** مبنية باستخدام Streamlit
- 🚀 **سرعة عالية** باستخدام ONNX Runtime
- 🎨 **معالجة ذكية للصور** (قص 18% فوق + 6% تحت + Median Filter)
- 📊 **عرض نتائج تفاعلي** مع رسوم بيانية
- 🔬 **عرض مراحل المعالجة** خطوة بخطوة
- 🩺 **توصيات طبية** حسب كل فئة
- 💾 **دقة عالية** تصل إلى 94.29%

---

## 📊 **أداء النموذج**

| المقياس                     | القيمة   |
| :-------------------------- | :------- |
| **دقة الاختبار**            | 94.29%   |
| **أفضل دقة تحقق**           | 96.40%   |
| **دقة MI (النوبة القلبية)** | 100%     |
| **دقة Abnormal**            | 100%     |
| **زمن التنبؤ**              | ~50ms    |
| **حجم النموذج**             | 27.76 MB |

---

## 📁 **هيكل المشروع**
