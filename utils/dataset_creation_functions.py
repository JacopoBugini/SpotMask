import os
import shutil


# Get files in a defined dir
def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))

    print("*****************************************")
    print(len(r), 'files found')
    print(dir)
    print("*****************************************")

    return r

# Obtain destination folder based on file names
def get_destination_folder(file, images_destination_folder):

    if "Mask_Nose_Mouth" in file:
        dst = os.path.join(images_destination_folder, 'Chin')
    elif "Mask_Mouth_Chin" in file:
        dst = os.path.join(images_destination_folder,'Nose')
    elif "Mask_Chin" in file:
        dst = os.path.join(images_destination_folder,'Off')
    elif "Mask" in file:
        dst = os.path.join(images_destination_folder,'On')
    else:
        dst = os.path.join(images_destination_folder,'Other')
    
    return dst

# Classify in destination folder
def classify_images(r):

    print(len(r), 'files in the directory')
    print("*****************************************")
    print("STARTING ALLOCATION")
    print("*****************************************")  
    
    count=0
    for file in r: 
        destination_folder = get_destination_folder(file)
        shutil.copy(file, destination_folder)
        count+=1
        print(count, '/', len(r))

    print("*****************************************")
    print("DONE")
    print("*****************************************")

    return

# Split dataset in different sets
def dataset_splitter(dir, split=[0.7, 0.1, 0.2]):

    print("*****************************************")
    print("STARTING ALLOCATION")
    print("*****************************************")  

    files = list_files(dir)
    num = len(files)
    train = round(split[0] * num, 0)
    validation = round(split[1] * num, 0)
    test = num - validation - train
    print("Train images:", train)
    print("Validation images:", validation)
    print("Test images:", test)

    train_c = 0
    validation_c = 0
    test_c = 0
    for file in files:
        if train_c <= train:
            dst = os.path.join(dir, 'train')
            shutil.move(file, dst) 
            train_c += 1
        elif validation_c <= validation:
            dst = os.path.join(dir, 'validation')
            shutil.move(file, dst) 
            validation_c += 1
        elif test_c <= test:
            dst = os.path.join(dir, 'test')
            shutil.move(file, dst) 
            test_c += 1
        else:
            print("Some error occurred")
    
    print("*****************************************")
    print("DONE")
    print(os.path.join(dir, 'train'), ':', train_c)
    print(os.path.join(dir, 'validation'), ':', validation_c)
    print(os.path.join(dir, 'test'), ':', test_c)
    print("*****************************************")