import streamlit as st
from os import listdir
from os.path import isfile, join

import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import matplotlib.pyplot as plt
from PIL import Image

from models.create_fasterrcnn_model import create_model
from utils.annotations import inference_annotations
from utils.general import set_infer_dir
from utils.transforms import infer_transforms

## Variables
image_dir = './example_img/'
img_files = [f for f in listdir(image_dir) if isfile(join(image_dir, f))]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = 'fasterrcnn_resnet50_fpn_v2'	
model_weights_path = './serving_model/best_model.pth'
detection_threshold = 0.9 # Could have been made with a sidebar
classes = ['__background__',
		  'Su-35','C-130 Hercules', 'C-17 Globemaster', 'C-5 Galaxy', 'F-16',
			'Tu-160 Blackjack', 'E-3 Sentry', 'B-52', 'P-3 Orion', 'B-1 Lancer',
			'E-135', 'Tu-22', 'F-15 Eagle', 'KC - xxx', 'F-22 Raptor',
			'F-18', 'Tu-90 Bear', 'KC-135', 'Su-35 Canard', 'Mig-23'
		]
num_classes = 21

st.title("Identification of planes on the ground.")

@st.cache
def load_model(model_weights_path, device, num_classes, model_type):
	checkpoint = torch.load(model_weights_path, map_location=device)

	build_model = create_model[model_type]
	model = build_model(num_classes=num_classes)
	model.load_state_dict(checkpoint['model_state_dict'])
	model.to(device).eval()
	return model

def inference(path):
	# To keep the annotation color the same each inference.
	np.random.seed(21)

	colors = np.random.uniform(0, 255, size=(len(classes), 3))

	#Get the image file name for saving output later on.
	image = cv2.imread(path)
	orig_image = image.copy()
	# BGR to RGB
	image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
	image = infer_transforms(image)
	# Add batch dimension.
	image = torch.unsqueeze(image, 0)
	start_time = time.time()
	with torch.no_grad():
		outputs = model(image.to(device))
		end_time = time.time()

	# Get the current ips.
	ips = 1 / (end_time - start_time)

	# Load all detection to CPU for further operations.
	outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
	# Carry further only if there are detected boxes.
	if len(outputs[0]['boxes']) != 0:
		orig_image = inference_annotations(
			outputs, detection_threshold, classes,
			colors, orig_image
		)
		image_bbox = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

		st.image(image_bbox, caption='Image with detection')


## Load model
model = load_model(model_weights_path, device, num_classes, model_type)	


# Select images

# text = st.text_area('Select one of the images provided or load your own', value="")
options = ['I will select from preloaded images', 'I will upload my own image']

col1, col2 = st.columns(2)

with col1:
	opt = st.radio('Please select the option for image input', options, index=0)
	img = None


with col2:
	calculate = st.button('Inference')


if opt == options[0]:
	img = st.selectbox('Select image', img_files)
	st.text('Please keep in mind that the model will resize your image to 640x640')
	image_path = image_dir +'/'+ img
elif opt ==options[1]:
	img_load = st.file_uploader('Please upload your image', disabled=False, label_visibility="visible")
	if img_load is not None:
		file_details = {"FileName":img_load.name,"FileType":img_load.type}
		# st.write(file_details)
		img = img_load
		image_path = os.path.join("example_img/tmp",img_load.name)
		with open(image_path,"wb") as f: 
			f.write(img_load.getbuffer())         
		image_path = os.path.join("example)_img/tmp",img_load.name)

col1, col2 = st.columns(2)

with col1:
	if img:
		image2show = Image.open(image_path)
		st.image(image2show, caption = 'Original')

with col2:
	
	if calculate:
	 	inference(image_path)

