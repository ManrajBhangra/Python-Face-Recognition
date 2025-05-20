from pathlib import Path

import face_recognition

import pickle

from collections import Counter

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

def recognize_faces(image_location: str, model: str = "hog", encodings_location = DEFAULT_ENCODINGS_PATH) -> None:
    with encodings_location.open(mode="rb") as i: #Load training set results
        loaded_encodings = pickle.load(i)
        
    input_image = face_recognition.load_image_file(image_location) #Load specified image
    
    input_face_locations = face_recognition.face_locations(input_image, model=model) #Get input face location
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations) #Get input face encodings
    
    for bound_box, unknown_encoding in zip(input_face_locations,input_face_encodings): #Loop through both locations and encodings simultaneously
        name = _recognize_face(unknown_encoding, loaded_encodings) #Find face if known name
        if not name: #Set default
            name = "Unknown"
        print(name, bound_box)
        
def _recognize_face(unknown_encoding, loaded_encodings):
    match = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding) #Find match
    
    votes = Counter(name for math, name in zip(match, loaded_encodings["names"]) if match) #Count matches
    
    if votes: #Return most common face
        return votes.most_common(1)[0][0]
    
if __name__ == "__main__":
    if not DEFAULT_ENCODINGS_PATH.exists(): #Check if model trained
        train()
        
recognize_faces("unknown.jpg")