import os
import shutil

#FaceDetectionImproperMask
images_original_directory = 'C:\\Users\\j.bugini\\Downloads\\FaceMask-Detection\\improper'
images_destination_folder = 'C:\\Users\\j.bugini\\Downloads\\FaceMask-Detection\\improper\\dataset'

train_folder = os.path.join(images_destination_folder, 'train')
validation_folder = os.path.join(images_destination_folder, 'validation')
test_folder = os.path.join(images_destination_folder, 'test')

print('Train folder is located at:', train_folder)
print('Validation folder is located at:', validation_folder)
print('Test folder is located at:', test_folder)

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


def get_destination_folder(file):

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



r=list_files('C:\\Users\\j.bugini\\Downloads\\FaceMask-Detection\\improper')
classify_images(r)
#dataset_splitter('C:\\Users\\j.bugini\\documents\\repos\\facemask-correctness-detection\\images\\Off')
#classify_images('C:\\Users\\j.bugini\\Downloads\\FaceMask-Detection\\FaceDetectionImproperMask')
