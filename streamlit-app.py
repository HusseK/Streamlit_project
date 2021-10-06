# -*- coding: utf-8 -*-
# @Author: Husseine Madany KEITA
# @Date:   2021-08-11 13:45:39
# @Last Modified by:   Husseine Madany Keita
# @Last Modified Date: 2021-08-13 17:54:34
try:
    import streamlit as st
    import streamlit.components.v1 as components
    import cv2
    import matplotlib.pyplot as plt
    import os
    from io import StringIO,BytesIO
    import numpy as np
    from PIL import Image

except:
    print("Il manque des modules.")

@st.cache(suppress_st_warning=True)
def boundig_boxes_on_image(img,overlap_thr = 0.3):
    #cwd="C:\Users\hkeita\Documents\Projet Streamlit\Streamlit_project"
    #model_path ="Dossier YOLO/models/darknet"    #"..\Dossier YOLO\models\darknet\"
    #label_path = os.path.join(model_path, "data", "coco.names")
    LABELS = open('coco.names').read().strip().split("\n")
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
    #config_path = os.path.join(model_path, "cfg", "yolov3.cfg")
    #weights_path = os.path.join(model_path, "yolov3.weights")
    (image_height, image_width) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                             swapRB=True, crop=False)
    net = cv2.dnn.readNetFromDarknet('yolov3.cfg','yolov3.weights')
    layers_names = net.getLayerNames()
    yolo_layers_names = [layers_names[iLayer[0] - 1] for iLayer in net.getUnconnectedOutLayers()]
    net.setInput(blob)
    yolo_layer_outputs = net.forward(yolo_layers_names)

    confidence_thr = 0.5

    boxes = []
    confidences = []
    classIDs = []

    # --- On parcoure les trois couches YOLO
    for iLayer, output in enumerate(yolo_layer_outputs):

        # --- On parcoure toutes les détections, par couche
        for iOutput, detection in enumerate(output):
            
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # --- On ne garde que les bounding boxes pour lesquelles la valeur de confiance est suffisamment élevée
            if confidence > confidence_thr:
                # print(iLayer, iOutput)
                
                # --- On réexprime les coordonnées des bounding boxes
                box = detection[0:4] * np.array([image_width, image_height, image_width, image_height])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thr, overlap_thr)
    if isinstance( idxs,tuple):
        return "Aucun objet n'a été détecté dans cette image"
    for i in idxs.flatten():

        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
#    st.write('''<style>
#                 body { margin: 0; font-family: Arial, Helvetica, sans-serif;} 
#                 .header{padding: 10px 16px; background: #555; color: #f1f1f1; position:fixed;top:0;} 
#                .sticky { position: fixed; top: 0; width: 100%;} 
#                </style>
#                <div class="header" id="myHeader">'+str(average_score)+'
#                </div>''', unsafe_allow_html=True
#            )
    
    components.html(
        '''
        <!DOCTYPE html>

        <html>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
            <link href="https://fonts.googleapis.com/css?family=Raleway:100,100i,200,200i,300,300i,400,400i,500,500i,600,600i,700,700i,800,800i,900,900i" rel="stylesheet">
            <link href="https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css" rel="stylesheet" integrity="sha384-wvfXpqpZZVQGK6TAh5PVlGOfQNHSoD2xbE+QkPxCAFlNEevoEH3Sl0sibVcOQVnN" crossorigin="anonymous">
            <link rel="stylesheet" type="text/css" href="https://cdnjs.cloudflare.com/ajax/libs/animate.css/3.5.2/animate.css">
            <link rel="stylesheet" type="text/css" href="css/main.css">
            <link href="https://fonts.googleapis.com/css?family=Alegreya+Sans:100,100i,300,300i,400,400i,500,500i,700,700i,800,800i,900,900i" rel="stylesheet">
        
        
        
        
              <head>
                  <header>
                
                      <nav class="navbar navbar-expand-lg navbar-light bg-light fixed-top" id="mainNav">
                      <div class="container-fluid">
                        <a class="navbar-brand js-scroll-trigger" href="https://positivethinking.tech/fr/">
                         <img src="https://i.postimg.cc/FH2Cf1sN/logo-ptc-techlab2.png", style="width: 15vw; min-width: 3px;padding-bottom: 3px;" > 
                        </a>
                
                
                
                        <div class="header">
                          <div class="progress-container">
                            <div class="progress-bar" id="myBar"></div>
                          </div>  
                        </div>
                
                        <div class="collapse navbar-collapse" id="navbarResponsive">
                          <ul class="navbar-nav ml-auto">
                            <li class="nav-item active">
                              <a class="nav-link" href="index.html">HOME <span class="sr-only">(current)</span></a>
                            </li>
                             <li class="nav-item dropdown  main-menu">
                            <li class="nav-item">
                              <a class="nav-link" href="https://positivethinking.tech/fr/contact/">CONTACT US</a>
                            </li>

                          </ul>
                        </div>
                      </div>
                
                    </nav>
                    </header>
            
              
                <meta charset="utf-8" />
                <meta name="viewport" content="width=device-width, initial-scale=1.0" /><meta name="generator" content="Docutils 0.17: http://docutils.sourceforge.net/" />
            
                <link rel="stylesheet" type="text/css" href="_static/pygments.css" />
                <link rel="stylesheet" type="text/css" href="_static/alabasterhusse.css" />
                <script data-url_root="./" id="documentation_options" src="_static/documentation_options.js"></script>
                <script src="_static/jquery.js"></script>
                <script src="_static/underscore.js"></script>
                <script src="_static/doctools.js"></script>
                
              <link rel="stylesheet" href="_static/custom.css" type="text/css" />
              
                <title>Untitled</title>
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.1.3/css/bootstrap.min.css">
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/ionicons/2.0.1/css/ionicons.min.css">
                <link rel="stylesheet" href="assets/css/style.css">
                          
              
              <meta name="viewport" content="width=device-width, initial-scale=0.9, maximum-scale=0.9" />
            
              </head>
              <body>
                  <script>
                    // When the user scrolls the page, execute myFunction 
                    window.onscroll = function() {myFunction()};
                    
                    function myFunction() {
                      var winScroll = document.body.scrollTop || document.documentElement.scrollTop;
                      var height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
                      var scrolled = (winScroll / height) * 100;
                      document.getElementById("myBar").style.width = scrolled + "%";
                    }
                    </script>
              <\body>
        ''',
        height=200)


    st.write('''
    # Bienvenue
    App simple pour la détection d'objets dans une image
    ''')
    show_image=st.empty()

    uploaded_file = st.file_uploader("Pick a file", type=['png','jpg','jpeg','bmp','pgm','tiff'])

    if not uploaded_file:
        show_image.info("Veuillez déposer une image s'il vous plait")
        return


    content = uploaded_file.getvalue()

    if isinstance(uploaded_file, BytesIO):
        show_image.image(uploaded_file)

    show_result=st.empty()
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    res=boundig_boxes_on_image(img_array)
    if isinstance(res, str):
        show_result.info(res)
        
        return
    else:
        show_result.image(res)

main()
