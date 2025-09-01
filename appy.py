import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
import os

# Configurações da página
st.set_page_config(
    page_title="Classificador de Lojas - Renner",
    page_icon="🏪",
    layout="wide"
)

st.title("🏪 Classificador de Lojas - Renner")
st.write("### Faça upload de uma imagem para classificação")

# Cache otimizado para a nuvem
@st.cache_resource
def load_your_model():
    try:
        # CAMINHO RELATIVO (importante para a cloud)
        model_path = 'modelo_renner.h5'
        
        # Verifica se o arquivo existe
        if not os.path.exists(model_path):
            st.error(f"❌ Arquivo do modelo não encontrado: {model_path}")
            return None
            
        model = load_model(model_path)
        st.sidebar.success("✅ Modelo carregado com sucesso!")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Erro ao carregar modelo: {str(e)}")
        return None

# Função de pré-processamento
def preprocess_for_your_model(img):
    try:
        target_size = (224, 224)
        img = img.resize(target_size)

        # Converte para array
        img_array = np.array(img)

        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]

        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        st.error(f"Erro no pré-processamento: {str(e)}")
        return None

# Suas classes
YOUR_CLASSES = [
    "RE",
    "Itaguacu"
]

# Carrega o modelo
model = load_your_model()

# Upload da imagem
uploaded_file = st.file_uploader(
    "📤 Selecione uma imagem de loja",
    type=['jpg', 'jpeg', 'png', 'webp'],
    help="Imagens de fachada ou interior de lojas"
)

if uploaded_file is not None:
    # Salva temporariamente
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Abre a imagem
        img = Image.open(tmp_path)

        # Layout em colunas
        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Imagem Carregada", use_column_width=True)
            st.success("✅ Imagem pronta para classificação")

            # Informações da imagem
            st.write(f"Dimensões: {img.size[0]}x{img.size[1]} pixels")
            st.write(f"Modo: {img.mode}")

        with col2:
            st.subheader("🔍 Classificação")

            if model is None:
                st.warning("⚠️ Modelo não carregado. Verifique se o arquivo 'modelo_renner.h5' está na pasta raiz.")
            else:
                if st.button("🎯 Executar Classificação", type="primary", use_container_width=True):
                    with st.spinner("Processando imagem..."):
                        # Pré-processa
                        processed_img = preprocess_for_your_model(img)

                        if processed_img is not None:
                            # Faz a predição
                            predictions = model.predict(processed_img)

                            # Obtém resultados
                            predicted_class_idx = np.argmax(predictions[0])
                            confidence = predictions[0][predicted_class_idx]
                            predicted_class = YOUR_CLASSES[predicted_class_idx]

                            # Exibe resultados
                            st.success("✅ Classificação concluída!")

                            # Métrica principal
                            st.metric(
                                label="Predição",
                                value=predicted_class,
                                delta=f"{confidence:.2%} confiança"
                            )

                            # Todas as probabilidades
                            st.subheader("📊 Probabilidades por classe:")
                            for i, (classe, prob) in enumerate(zip(YOUR_CLASSES, predictions[0])):
                                progress_value = float(prob)
                                st.write(f"{classe}: {prob:.4f} ({prob:.2%})")
                                st.progress(progress_value, text=f"{prob:.2%}")

    except Exception as e:
        st.error(f"❌ Erro: {str(e)}")

    finally:
        # Limpa arquivo temporário
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

else:
    st.info("👆 Aguardando upload de imagem...")
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://via.placeholder.com/400x300?text=Faça+upload+da+imagem+da+loja",
                 caption="Exemplo: imagem de fachada ou interior", width=350)
    with col2:
        st.write("""
        ### 📝 Instruções:
        1. Clique em 'Browse files' ou arraste uma imagem
        2. A imagem deve ser de uma loja (fachada ou interior)
        3. Clique em 'Executar Classificação'
        4. Veja os resultados da classificação
        """)

# Sidebar com informações
with st.sidebar:
    st.header("⚙️ Configurações do Modelo")
    st.write(f"Classes configuradas: {len(YOUR_CLASSES)}")
    for i, classe in enumerate(YOUR_CLASSES):
        st.write(f"{i+1}. {classe}")

    st.header("📋 Requisitos")
    st.write("""
    - Formatos: JPG, JPEG, PNG
    - Tamanho recomendado: > 300x300 pixels
    - Imagens claras e bem enquadradas
    """)