from pathlib import Path

import face_recognition

import pickle

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl") #Default location to store file

def train(model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH) ->None:
    names = [] #Store names
    encodings = [] #Store encodings of face
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name #Get name of face from folder
        image = face_recognition.load_image_file(filepath) #Load Image
        
        face_locations = face_recognition.face_locations(image, model=model) #Find the face on the image
        face_encodings = face_recognition.face_encodings(image, face_locations) #Extract the encodings from image
        
        for encoding in face_encodings: #Append to respective lists
            names.append(name)
            encodings.append(encoding)
    
    name_encodings = {"names": names, "encodings": encodings} #Create dictionary to store both names and encodings together
    
    with encodings_location.open(mode="wb") as i: #Save the file using pickle
        pickle.dump(name_encodings, i)

train()