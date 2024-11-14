from threading import Thread
import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import plotly.graph_objects as go
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
import google.generativeai as genai
import PIL.Image
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.getenv("GROQ_API_KEY"))

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

output_dir = 'saliency_maps'
os.makedirs(output_dir, exist_ok=True)

def generate_explanation(img_path, model_prediction, confidence, gen_model):

  prompt = f""" MRI Analysis System Prompt
  You are an expert neuroradiologist with extensive experience in tumor diagnosis and medical imaging interpretation. Your task is to analyze and explain the relationship between an MRI scan's saliency map and its automated classification results.
  Context Parameters

  Primary Image: Brain MRI scan
  Analysis Type: Saliency map interpretation
  Model Prediction: {model_prediction}
  Confidence Score: {confidence * 100}%
  Possible Classifications: Glioma, Meningioma, Pituitary, No Tumor

  Required Analysis Components
  1. Anatomical Analysis

  Identify specific brain structures and regions highlighted in the saliency map
  Reference standard anatomical landmarks
  Note spatial relationships to critical brain structures
  Describe tumor location using standardized anatomical terminology

  2. Classification Justification

  Correlate highlighted regions with typical presentation patterns for the predicted tumor type
  Compare findings against known radiological features of the predicted class
  Explain why these features support the model's classification
  Address any atypical presentations if present

  3. Clinical Context

  Discuss the typical characteristics of the predicted tumor type
  Note common symptoms associated with tumors in the highlighted locations
  Reference relevant prognostic implications
  Suggest appropriate follow-up imaging or diagnostic procedures

  4. Technical Assessment

  Evaluate the strength of the model's prediction based on:

  Confidence score
  Anatomical accuracy of highlighted regions
  Consistency with clinical presentation patterns


  Note any technical limitations or artifacts

  <Output Format>

  **Radiological Analysis Report**

  **Anatomical Findings:**
  [2-3 sentences describing the specific brain regions highlighted and their anatomical significance]

  **Classification Analysis:**
  [2-3 sentences explaining how the highlighted regions support the model's classification]

  **Clinical Implications:**
  [2-3 sentences discussing the medical significance and recommended next steps]

  **Technical Assessment:**
  [1-2 sentences evaluating the reliability of the model's prediction]

  **Key Features Supporting Classification:**
  - [Bullet point 1]
  - [Bullet point 2]
  - [Bullet point 3]

  **Recommended Follow-up:**
  [1-2 specific recommendations for clinical validation or further testing]
  </Output Format>

  Response Guidelines

  Maintain professional medical terminology while ensuring clarity
  Support observations with anatomical reasoning
  Be specific about location and extent of highlighted regions
  Connect anatomical findings to clinical significance
  Maintain appropriate clinical uncertainty where warranted
  Avoid technical discussion of the AI model's architecture or general saliency map properties
  Focus on medical interpretation rather than technical aspects
  Include clear, actionable next steps for clinical validation

  Constraints

  Keep each section concise and focused
  Use standard medical terminology with brief clarifications where needed
  Avoid speculation about patient-specific outcomes
  Maintain focus on radiological findings rather than treatment planning
  Acknowledge limitations of AI-based analysis when appropriate

  Please verify step by step.
  """

  img = PIL.Image.open(img_path)

  if(gen_model == "Gemini"):
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content([prompt, img])
    return response.text
  else:
    raw_response = client.chat.completions.create(model="llama-3.2-3b-preview",
                                                    messages=[{
                                                        "role": "user",
                                                        "content": prompt
                                                    }])
    return raw_response.choices[0].message.content



def generate_saliency_map(model, img_array, class_index, img_size):
  with tf.GradientTape() as tape:
    img_tensor = tf.convert_to_tensor(img_array)
    tape.watch(img_tensor)
    predictions = model(img_tensor)
    target_class = predictions[:, class_index]

  gradients = tape.gradient(target_class, img_tensor)
  gradients = tf.math.abs(gradients)
  gradients = tf.reduce_max(gradients, axis=-1)
  gradients = gradients.numpy().squeeze()

  # Resize gradients to match original image size
  gradients = cv2.resize(gradients, img_size)

  # Create a circular mask for the brain area
  center = (gradients.shape[0] // 2, gradients.shape[1] // 2)
  radius = min(center[0], center[1]) - 10
  y, x = np.ogrid[:gradients.shape[0], :gradients.shape[1]]
  mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2

  # Apply mask to gradients
  gradients = gradients * mask

  # Normalize only the brain area
  brain_gradients = gradients[mask]
  if brain_gradients.max() > brain_gradients.min():
    brain_gradients = (brain_gradients - brain_gradients.min()) / (brain_gradients.max() - brain_gradients.min())
  gradients[mask] = brain_gradients

  # Apply a higher threshold
  threshold = np.percentile(gradients[mask], 80)
  gradients[gradients < threshold] = 0

  # Apply more aggressive smoothing
  gradients = cv2.GaussianBlur(gradients, (11, 11), 0)

  # Create a heatmap overlay with enhanced contrast
  heatmap = cv2.applyColorMap(np.uint8(255 * gradients), cv2.COLORMAP_JET)
  heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

  # Resize heatmap to match original image size
  heatmap = cv2.resize(heatmap, img_size)

  # Superimpose the heatmap on original image with increased opacity
  original_img = image.img_to_array(img)
  superimposed_img = heatmap * 0.7 + original_img * 0.3
  superimposed_img = superimposed_img.astype(np.uint8)

  img_path = os.path.join(output_dir, uploaded_file.name)
  with open(img_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

  saliency_map_path = f"saliency_maps/{uploaded_file.name}"

  # Save the saliency map
  cv2.imwrite(saliency_map_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

  return superimposed_img

def load_xception_model(model_path):
  img_shape=(299, 299, 3)
  base_model = tf.keras.applications.Xception(include_top=False, weights="imagenet",
                                              input_shape=img_shape, pooling='max')

  model = Sequential([
      base_model,
      Flatten(),
      Dropout(rate=0.3),
      Dense(128, activation='relu'),
      Dropout(rate=0.25),
      Dense(4, activation='softmax')
  ])

  model.build((None,) + img_shape)

  # Compile the model
  model.compile(Adamax(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy',
                         Precision(),
                         Recall()])

  model.load_weights(model_path)

  return model

st.title("Brain Tumor Classification")

st.write("Upload an image of a brain MRI scan to classify.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

  selected_model = st.radio(
    "Select Model",
    ("Transfer Learning - Xception", "Custom Convolutional Neural Network")
  )

  if selected_model == "Transfer Learning - Xception":
    model = load_xception_model('xception_model.weights.h5')
    img_size = (299, 299)
  else:
    model = load_model('cnn_model.h5')
    img_size = (224, 224)

  labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
  img = image.load_img(uploaded_file, target_size=img_size)
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  img_array /= 255.0


  with st.spinner('Generating Prediction...'):
    prediction = model.predict(img_array)

  # Get the class with the highest probability
  class_index = np.argmax(prediction[0])
  result = labels[class_index]

  st.write(f"Prediction: {result}")
  st.write("Prediction:")
  for label, prob in zip(labels, prediction[0]):
    st.write(f"{label}: {prob:.4f}")

  with st.spinner('Generating saliency map...'):
    saliency_map = generate_saliency_map(model, img_array, class_index, img_size)

  col1, col2 = st.columns(2)
  with col1:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
  with col2:
    st.image(saliency_map, caption="Saliency Map", use_container_width=True)

  st.write('## Classification Results')

  result_container = st.container()
  result_container.markdown(
      f"""
      <div style="background-color: #000000; color: #ffffff; paddingL 30px; border-radius: 15px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <div style="flex: 1; text-align: center;">
            <h3 style="color: #ffffff; margin-bottom: 10px; font-size: 20px;">Prediction</h3>
            <p style="font-size: 36px; font-weight: 800; color: #FF0000; margin: 0;">
              {result}
            </p>
          </div>
          <div style="width: 2px; height: 80px; background-color: #ffffff; margin: 0 20px;"></div>
            <h3 style="color: #ffffff; margin-bottom: 10px; font-size: 20px;">Confidence</h3>
            <p style="font-size: 36px; font-weight: 800; color: #2196F3; margin: 0">
              {prediction[0][class_index]:.4%}
            </p>
          </div>
        </div>
      </div>
      """,
      unsafe_allow_html=True
  )

  # Prepare data for Plotly chart
  probabilities = prediction[0]
  sorted_indices = np.argsort(probabilities)
  sorted_labels = [labels[i] for i in sorted_indices]
  sorted_probabilities = probabilities[sorted_indices]

  # Create a Plotly bar chart
  fig = go.Figure(go.Bar(
    x=sorted_probabilities,
    y=sorted_labels,
    orientation='h',
    marker_color=['red' if label == result else 'blue' for label in sorted_labels]
  ))

  # Customize the chart layout
  fig.update_layout(
    title='Probabilities for each class',
    xaxis_title='Probability',
    yaxis_title='Class',
    height=400,
    width=600,
    yaxis=dict(autorange="reversed")
  )

  # Add value labels to the bars
  for i, prob in enumerate(sorted_probabilities):
    fig.add_annotation(
        x=prob,
        y=i,
        text=f"{prob:.4f}",
        showarrow=False,
        xanchor='left',
        xshift=5
    )
  # Display the Plotly chart
  st.plotly_chart(fig)

  saliency_map_path = f"saliency_maps/{uploaded_file.name}"



  selected_gen_model = st.radio(
    "Select Generative Model for Explanation",
    ("Gemini 1.5 Flash", "Llama 3.2-3b Preview")
  )

  with st.spinner('Generating explanation...'):
        if selected_gen_model == "Gemini 1.5 Flash":
            explanation = generate_explanation(saliency_map_path, result, prediction[0][class_index], "Gemini")
        else:
            explanation = generate_explanation(saliency_map_path, result, prediction[0][class_index], "Llama")

        st.write("## Explanation")
        st.write(explanation)
