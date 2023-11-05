import streamlit as st
import tensorflow as tf
import os
import tensorflow_hub as hub
from styler import load_img, transform_img, tensor_to_image, imshow

st.set_page_config(page_title="AestheTech", layout="wide")
st.write("""
# AestheTech: Generate artistic images with AI!
""")

# Loading Pretrained Model from TFHub
def load_model():
    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    return model


model_load_state = st.text('The canvas awaits, a brushstroke away')
model = load_model()
model_load_state.text('')


content_image, style_image = st.columns(2,gap="large")
result_image, _ =st.columns(2)

# upload a content image
with content_image:
    st.write('***')
    st.write('### Content Image')
    content_image_file = st.file_uploader(
        "Upload a Content image (i.e. the image on which you want to apply the style)", type=("png", "jpg", "jpeg"))
    try:
        content_image_file = content_image_file.read()
        content_image_file = transform_img(content_image_file)
    except:
        pass
        

# display the content image
    try:
        st.image(imshow(content_image_file), width=500)
        st.write('***')
    except:
        pass


# upload a style image
with style_image:
    st.write('***')
    st.write('### Style Image')
    chosen_style = "Upload"
    style_image_file = st.file_uploader(
        "Upload a Style image (i.e. the style/pattern/texture that you want your new image to have)", type=("png", "jpg", "jpeg"))
    try:
        style_image_file = style_image_file.read()
        style_image_file = transform_img(style_image_file)
    except:
        pass

# display the style image
    try:
        st.image(imshow(style_image_file), width=500)
        st.write('***')
    except:
        pass


# result image
with result_image:
    st.write('### Result Image')
    button_style = """
            <style>
            .stButton > button {
                color: red;
                background: white;
                border: 2px solid;
                border-radius: 5px;
                font-size: 200px;
                width: 150px;
                height: 55px;
            }
            </style>
            """
    st.markdown(button_style, unsafe_allow_html=True)

    predict = st.button("***Generate***")
    with st.spinner("Blending content + style..."):
        st.write('\n')
        if predict:
            if content_image_file is not None and style_image_file is not None:
                try:
                    stylized_image = model(tf.constant(content_image_file), tf.constant(style_image_file))[0]
                    generated_image = tensor_to_image(stylized_image)
                except:
                    stylized_image = model(tf.constant(tf.convert_to_tensor(content_image_file[:, :, :, :3])),tf.constant(tf.convert_to_tensor(style_image_file[:, :, :, :3])))[0]
                    generated_image = tensor_to_image(stylized_image)

                st.image(generated_image)
               
                try:
                    # Delete style.jpg and content.jpg
                    os.remove("style.jpg")
                    os.remove("content.jpg")
                except:
                    pass
            else:
                st.markdown("#### Please upload both the images to proceed!")


