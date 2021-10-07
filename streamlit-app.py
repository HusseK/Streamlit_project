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
        return "Aucun objet n'a été détecté dans cette image",0
    for i in idxs.flatten():

        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), idxs.shape[0]


def main():

    st.sidebar.image("logo_ptc_techlab.png", width = 150)#, use_column_width=True)



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
    res,nb=boundig_boxes_on_image(img_array)
    if isinstance(res, str):
        show_result.info(res)
        
        return
    else:
        show_result.image(res)
    show_info = st.empty()
    show_info.info("Nombre d'objets détectés: "+str(nb))

main()
