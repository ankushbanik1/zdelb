import face_recognition
import cv2
import pandas as pd
import numpy as np
from scipy.spatial import distance
import operator
from scipy.stats import pearsonr


def recommend(user_img, data):
    """
        Takes in user image and dataframe consisting image details in database

        Returns five image paths
    """
    #encoding user image
    image = cv2.imread(user_img)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="cnn")
    user_encodings = face_recognition.face_encodings(rgb, boxes)

    #checking all distances
    dist = []
    for i in range(data.shape[0]):
        enc = data['image_encodings'][i]
        dist.append(distance.cosine(enc, user_encodings))
    data['distances'] = pd.Series(dist)
    
    data = data.sort_values('distances').head(50)
    final_images = data['FilePath'].tolist()

    
    #user image array
    IMG_SIZE=70
    img = cv2.imread(user_img, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    length = np.prod(new_array.shape)
    new_array = new_array.reshape(length)

    #applying cosine score
    diff = {}
    for img in final_images:
      img1 = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
      new_array1 = cv2.resize(img1, (IMG_SIZE, IMG_SIZE))
      length = np.prod(new_array1.shape)
      new_array1 = new_array1.reshape(length)
      diff[img] = abs(distance.cosine(new_array1, new_array))
        
    sorted_diff = dict(sorted(diff.items(), key=operator.itemgetter(1)))
    final = list(sorted_diff.keys())
    final_cosine = final[:35]

    #applying jaccard score
    diff = {}
    for img in final_images:
      img1 = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
      new_array1 = cv2.resize(img1, (IMG_SIZE, IMG_SIZE))
      length = np.prod(new_array1.shape)
      new_array1 = new_array1.reshape(length)
      diff[img] = abs(distance.jaccard(new_array1, new_array))
        
    sorted_diff = dict(sorted(diff.items(), key=operator.itemgetter(1)))
    final = list(sorted_diff.keys())
    final_jaccard = final[:35]

    #applying pearson correlation score
    diff = {}
    for img in final_images:
      img1 = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
      new_array1 = cv2.resize(img1, (IMG_SIZE, IMG_SIZE))
      length = np.prod(new_array1.shape)
      new_array1 = new_array1.reshape(length)
      diff[img] = abs(pearsonr(new_array1, new_array)[0])
        
    sorted_diff = dict(sorted(diff.items(), key=operator.itemgetter(1)))
    final = list(sorted_diff.keys())
    final_pearson = final[-35:]

    #selecting 5 images
    ultimate = []
    for i in final_cosine:
      if i in final_jaccard:
        if i in final_pearson[::-1]:
          ultimate.append(i)

    return ultimate[:5]
    
