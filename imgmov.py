import streamlit as st
import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
import tempfile
import gdown
from tqdm import tqdm
import requests

def download_model_weights():
    """
    تحميل ملف النموذج مع دعم مصادر متعددة
    """
    output = 'vox-cpk.pth.tar'
    
    if os.path.exists(output):
        return output
        
    # قائمة روابط بديلة للنموذج
    urls = [
        'https://drive.google.com/uc?id=1PyQJmkdCsAkOYwUyaj_l-l0as-iLDgeH',  # الرابط الأصلي
        'https://huggingface.co/datasets/mishig/first-order-motion-model/resolve/main/vox-cpk.pth.tar',  # رابط بديل على Hugging Face
    ]
    
    for url in urls:
        try:
            st.info(f"جاري محاولة تحميل النموذج من {url}")
            
            if 'drive.google.com' in url:
                try:
                    gdown.download(url, output, quiet=False)
                    if os.path.exists(output):
                        st.success("تم تحميل النموذج بنجاح!")
                        return output
                except Exception as e:
                    st.warning(f"فشل التحميل من Google Drive: {str(e)}")
                    continue
            else:
                # استخدام requests للروابط المباشرة
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                progress_bar = st.progress(0)
                progress_text = st.empty()
                
                with open(output, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            progress = int((downloaded / total_size) * 100)
                            progress_bar.progress(progress / 100)
                            progress_text.text(f"تم تحميل {progress}%")
                
                if os.path.exists(output):
                    st.success("تم تحميل النموذج بنجاح!")
                    return output
                
        except Exception as e:
            st.warning(f"فشل التحميل من {url}: {str(e)}")
            continue
    
    raise Exception("فشل تحميل النموذج من جميع المصادر المتاحة")

def load_model():
    model_weights = download_model_weights()
    if not os.path.exists('first_order_model'):
        os.system('git clone https://github.com/AliaksandrSiarohin/first-order-model.git first_order_model')
    
    sys.path.append('first_order_model')
    from demo import load_checkpoints, make_animation
    
    generator, kp_detector = load_checkpoints(config_path='first_order_model/config/vox-256.yaml', 
                                            checkpoint_path=model_weights)
    return generator, kp_detector, make_animation

def process_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.resize((256, 256))
    image = np.array(image)
    return image

@st.cache_resource
def get_model():
    return load_model()

def main():
    st.title("تحريك الصور باستخدام First Order Motion Model")
    
    # إضافة خيار لتحميل النموذج يدوياً
    uploaded_model = st.file_uploader("(اختياري) ارفع ملف النموذج يدوياً (vox-cpk.pth.tar)", type=['tar'])
    if uploaded_model:
        with open('vox-cpk.pth.tar', 'wb') as f:
            f.write(uploaded_model.getbuffer())
        st.success("تم رفع ملف النموذج بنجاح!")
    
    # تحميل النموذج
    try:
        generator, kp_detector, make_animation = get_model()
        st.success("تم تحميل النموذج بنجاح!")
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل النموذج: {str(e)}")
        st.info("يمكنك محاولة تحميل ملف النموذج يدوياً من أحد الروابط التالية:")
        st.markdown("""
        - [تحميل من Hugging Face](https://huggingface.co/datasets/mishig/first-order-motion-model/resolve/main/vox-cpk.pth.tar)
        - [تحميل من Google Drive](https://drive.google.com/uc?id=1PyQJmkdCsAkOYwUyaj_l-l0as-iLDgeH)
        """)
        return

    # رفع الصورة الثابتة
    source_image = st.file_uploader("ارفع الصورة الثابتة", type=['jpg', 'jpeg', 'png'])
    
    # رفع فيديو الحركة
    driving_video = st.file_uploader("ارفع فيديو مصدر الحركة", type=['mp4', 'avi', 'mov'])

    if source_image and driving_video:
        try:
            # معالجة الصورة المصدر
            source_image = Image.open(source_image)
            source_image = process_image(source_image)
            
            # حفظ الفيديو مؤقتاً
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(driving_video.read())
            
            # قراءة الفيديو
            cap = cv2.VideoCapture(tfile.name)
            driving_frames = []
            
            progress_bar = st.progress(0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                driving_frames.append(process_image(frame))
                progress_bar.progress((i + 1) / frame_count)
            
            cap.release()
            os.unlink(tfile.name)
            
            if len(driving_frames) > 0:
                with st.spinner('جاري إنشاء الرسوم المتحركة...'):
                    predictions = make_animation(source_image, driving_frames, generator, kp_detector, 
                                              relative=True, adapt_movement_scale=True)
                
                st.video(predictions)
                st.success("تم إنشاء الرسوم المتحركة بنجاح!")
            else:
                st.error("لم يتم العثور على إطارات في الفيديو")
                
        except Exception as e:
            st.error(f"حدث خطأ أثناء معالجة الصور: {str(e)}")

if __name__ == "__main__":
    main()