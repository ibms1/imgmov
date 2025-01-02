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

def download_model_weights():
    # رابط نموذج FOMM المدرب مسبقاً
    url = 'https://drive.google.com/uc?id=1PyQJmkdCsAkOYwUyaj_l-l0as-iLDgeH'
    output = 'vox-cpk.pth.tar'
    
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    return output

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
    
    # تحميل النموذج
    try:
        generator, kp_detector, make_animation = get_model()
        st.success("تم تحميل النموذج بنجاح!")
    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل النموذج: {str(e)}")
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
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                driving_frames.append(process_image(frame))
            
            cap.release()
            os.unlink(tfile.name)
            
            if len(driving_frames) > 0:
                # إنشاء الرسوم المتحركة
                predictions = make_animation(source_image, driving_frames, generator, kp_detector, 
                                          relative=True, adapt_movement_scale=True)
                
                # عرض النتيجة
                st.video(predictions)
                st.success("تم إنشاء الرسوم المتحركة بنجاح!")
            else:
                st.error("لم يتم العثور على إطارات في الفيديو")
                
        except Exception as e:
            st.error(f"حدث خطأ أثناء معالجة الصور: {str(e)}")

if __name__ == "__main__":
    main()