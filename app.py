# app.py - CUCUMBER DISEASE DETECTOR FOR STREAMLIT CLOUD
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="🥒 Cucumber Disease Detector",
    page_icon="🥒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CONFIGURATION ======================
MODEL_CONFIG = {
    'model_filename': 'cucumber_model.h5',  # Your model file in Streamlit Cloud
    'input_size': (224, 224),
    'classes': ['Downy_mildew', 'Healthy', 'Powdery_mildew'],
    'class_display': {
        'Downy_mildew': '🦠 Downy Mildew',
        'Healthy': '🌿 Healthy',
        'Powdery_mildew': '🍄 Powdery Mildew'
    }
}

# Disease information
DISEASE_INFO = {
    'Downy_mildew': {
        'description': 'Fungal disease caused by Pseudoperonospora cubensis',
        'symptoms': 'Yellow angular spots on upper leaf surface, grayish-purple mold on underside',
        'treatment': 'Apply copper-based fungicides or fosetyl-aluminum. Improve air circulation.',
        'severity': 'High',
        'prevention': 'Avoid overhead watering, use resistant varieties'
    },
    'Healthy': {
        'description': 'Healthy cucumber plant with no disease symptoms',
        'symptoms': 'Green, vibrant leaves without spots or discoloration',
        'treatment': 'Continue good agricultural practices',
        'severity': 'None',
        'prevention': 'Regular monitoring, balanced fertilization'
    },
    'Powdery_mildew': {
        'description': 'Fungal disease caused by Podosphaera xanthii',
        'symptoms': 'White powdery coating on leaves and stems',
        'treatment': 'Apply sulfur-based fungicides or potassium bicarbonate sprays',
        'severity': 'Medium',
        'prevention': 'Ensure good air circulation, avoid dense planting'
    }
}

# ====================== MODEL LOADING ======================
@st.cache_resource
def load_model():
    """Load the TensorFlow model from local file"""
    model_path = MODEL_CONFIG['model_filename']
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found!")
        st.info(f"""
        **To fix this:**
        1. Make sure 'cucumber_model.h5' is in your GitHub repository
        2. The file should be in the same directory as app.py
        3. File size should be around 2.6GB
        
        **Current files in directory:**
        {os.listdir('.')}
        """)
        return None
    
    try:
        # Load the model
        with st.spinner("🔄 Loading model into memory..."):
            model = tf.keras.models.load_model(model_path, compile=False)
            file_size = os.path.getsize(model_path) / (1024**3)
            st.success(f"✅ Model loaded successfully! ({file_size:.1f}GB)")
            return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# ====================== IMAGE PROCESSING ======================
def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Resize
    image = image.resize(MODEL_CONFIG['input_size'])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Normalize
    image_array = np.array(image) / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def predict_disease(model, image):
    """Make prediction on image"""
    # Preprocess
    processed_image = preprocess_image(image)
    
    # Predict
    predictions = model.predict(processed_image, verbose=0)[0]
    
    # Get results
    top_idx = np.argmax(predictions)
    disease = MODEL_CONFIG['classes'][top_idx]
    confidence = predictions[top_idx]
    
    # Get all probabilities
    all_probs = {
        MODEL_CONFIG['classes'][i]: float(predictions[i]) 
        for i in range(len(predictions))
    }
    
    return disease, confidence, all_probs

# ====================== UI COMPONENTS ======================
def display_disease_info(disease, confidence):
    """Display disease information card"""
    info = DISEASE_INFO[disease]
    
    st.markdown(f"""
    <div style="padding: 2rem; border-radius: 15px; margin: 1.5rem 0; 
                background: {'#d4edda' if disease == 'Healthy' else '#f8d7da'};
                border: 2px solid {'#28a745' if disease == 'Healthy' else '#dc3545'};
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h2 style="text-align: center; margin-bottom: 1rem; color: {'#28a745' if disease == 'Healthy' else '#dc3545'}">
            {'✅' if disease == 'Healthy' else '⚠️'} {disease.replace('_', ' ').upper() if disease != 'Healthy' else 'HEALTHY PLANT'}
        </h2>
        <h3 style="text-align: center; color: {'#28a745' if disease == 'Healthy' else '#dc3545'}">
            Confidence: {confidence*100:.1f}%
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Disease details
    st.markdown(f"""
    <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 10px; 
                margin: 1rem 0; border-left: 4px solid #6c757d;">
        <h4>📋 Disease Information</h4>
        <p><strong>Description:</strong> {info['description']}</p>
        <p><strong>Symptoms:</strong> {info['symptoms']}</p>
        <p><strong>Treatment:</strong> {info['treatment']}</p>
        <p><strong>Severity:</strong> {info['severity']}</p>
        <p><strong>Prevention:</strong> {info['prevention']}</p>
    </div>
    """, unsafe_allow_html=True)

def display_probability_chart(probabilities):
    """Display probability distribution"""
    st.subheader("📊 Probability Distribution")
    
    # Sort by probability
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    for disease, prob in sorted_probs:
        display_name = MODEL_CONFIG['class_display'][disease]
        
        # Create columns for layout
        col1, col2, col3 = st.columns([3, 5, 2])
        
        with col1:
            st.write(f"**{display_name}**")
        
        with col2:
            st.progress(float(prob))
        
        with col3:
            st.write(f"**{prob*100:.1f}%**")

# ====================== MAIN APP ======================
def main():
    st.title("🥒 Cucumber Disease Detector")
    st.markdown("Upload images of cucumber leaves to detect diseases instantly")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model status
        model_path = MODEL_CONFIG['model_filename']
        if os.path.exists(model_path):
            size_gb = os.path.getsize(model_path) / (1024**3)
            st.success(f"✅ Model Ready ({size_gb:.1f}GB)")
        else:
            st.warning("⚠️ Model not found in directory")
            st.write("**Files present:**")
            for file in os.listdir('.'):
                st.write(f"- {file}")
        
        st.header("🌿 Diseases Detected")
        for disease in MODEL_CONFIG['classes']:
            display_name = MODEL_CONFIG['class_display'][disease]
            st.write(f"{display_name}")
        
        st.header("📸 Image Guidelines")
        st.write("• Clear, well-lit leaf photos")
        st.write("• Leaf should fill most of frame")
        st.write("• Supported: JPG, PNG, JPEG")
        
        st.header("ℹ️ About")
        st.write("Model: CNN trained on 3,703 cucumber leaf images")
        st.write("Accuracy: ~96% (validation)")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a cucumber leaf image",
            type=['jpg', 'jpeg', 'png'],
            help="Select a clear image of a cucumber leaf"
        )
        
        if uploaded_file:
            try:
                # Load and display image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Image details
                with st.expander("📋 Image Details"):
                    st.write(f"**Filename:** {uploaded_file.name}")
                    st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
                    st.write(f"**Dimensions:** {image.size[0]} × {image.size[1]}")
                    
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
    
    with col2:
        st.subheader("🔍 Analysis")
        
        if uploaded_file:
            if st.button("🧪 Analyze Disease", type="primary", use_container_width=True):
                with st.spinner("Initializing model..."):
                    # Load model
                    model = load_model()
                    
                    if model:
                        try:
                            # Process image
                            image = Image.open(uploaded_file)
                            
                            # Make prediction
                            disease, confidence, all_probs = predict_disease(model, image)
                            
                            # Display results
                            display_disease_info(disease, confidence)
                            
                            # Show probabilities
                            display_probability_chart(all_probs)
                            
                            # Confidence metrics
                            st.subheader("📈 Confidence Metrics")
                            col_a, col_b, col_c = st.columns(3)
                            
                            with col_a:
                                st.metric("Top Confidence", f"{confidence*100:.1f}%")
                            
                            with col_b:
                                sorted_probs = sorted(all_probs.values(), reverse=True)
                                if len(sorted_probs) > 1:
                                    gap = (sorted_probs[0] - sorted_probs[1]) * 100
                                    st.metric("Margin", f"{gap:.1f}%")
                            
                            with col_c:
                                certainty = confidence * 100
                                if certainty > 90:
                                    st.metric("Certainty", "Very High")
                                elif certainty > 70:
                                    st.metric("Certainty", "High")
                                else:
                                    st.metric("Certainty", "Moderate")
                            
                        except Exception as e:
                            st.error(f"Prediction error: {str(e)}")
                    else:
                        st.error("Model failed to load. Check if 'cucumber_model.h5' is in the repository.")
            
            else:
                st.info("👆 Click 'Analyze Disease' to begin")
        
        else:
            st.info("👈 Upload an image to begin analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>🥒 Cucumber Disease Detection System</strong></p>
        <p>Model: 2.6GB CNN | Training: 3,703 images | Framework: TensorFlow</p>
        <p>Deployed on Streamlit Cloud</p>
        <p><em>⚠️ For agricultural guidance only. Consult experts for severe cases.</em></p>
    </div>
    """, unsafe_allow_html=True)

# ====================== RUN APP ======================
if __name__ == "__main__":
    main()