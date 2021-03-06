import numpy as np
import cv2
import os

class DatasetLoader:
    def __init__(self,preprocessors=None):
        self.preprocessors =preprocessors

        if self.preprocessors is None:
            self.preprocessors =[]

    def load(self,imagePaths,verbose=-1):
        data =[]
        labels=[]

        for(i, imagePath) in enumerate(imagePaths):
            if i == 0:
                print("imagepath: " + imagePath)
            image = cv2.imread(imagePath)
            if i == 0:
                print("label before: " + str(imagePath.split(os.path.sep)))
            label =imagePath.split(os.path.sep)[-2]

            if i == 0:
                print("label after: " + label)
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image =p.preprocess(image)
            data.append(image)
            labels.append(label)

            if verbose > 0 and (i+1) % verbose == 0:
                print("[INFO] processed {}/{}".format(i+1, len(imagePaths)))

        return(np.array(data),np.array(labels))


