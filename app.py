import streamlit as st
import tensorflow_hub as hub
from processing import bytes_to_tensor, apply_style, tensor_to_image


HUB_MODEL = hub.load(
    'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
)


st.set_page_config(layout='wide')
st.title('Style Transfer')
st.text(
    'This app uses a trained deep learning model to convert an image '
    '(the "Base Image") into the style of another (the "Style Image")'
)
col1, col2, col3 = st.columns(spec=3, gap='medium')


with col1:
    st.subheader('Style Image')
    style_file = st.file_uploader(
        label='Style Image',
        label_visibility='hidden',
        accept_multiple_files=False,
    )
    if style_file is not None:
        st.image(style_file)


with col2:
    st.subheader('Base Image')
    base_file = st.file_uploader(
        label='Base Image',
        label_visibility='hidden',
        accept_multiple_files=False,
    )
    if base_file is not None:
        st.image(base_file)


with col3:

    st.subheader('Output Image')
    st.text('')
    st.text('')

    clicked = st.button(label='Generate')
    st.text('')
    st.text('')
    st.text('')
    st.text('')
    st.text('')
    st.text('')

    if clicked and style_file and base_file:
        style_tensor = bytes_to_tensor(style_file.getvalue())
        base_tensor = bytes_to_tensor(base_file.getvalue())
        output_tensor = apply_style(base_tensor, style_tensor, HUB_MODEL)
        output_img = tensor_to_image(output_tensor)
        st.image(output_img)

    else:
        st.text('Please Upload Style and Base Image ...')
