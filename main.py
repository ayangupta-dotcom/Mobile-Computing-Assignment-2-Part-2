import cv2
import numpy as np
import os
import frameextractor as fe
from handshape_feature_extractor import HandShapeFeatureExtractor as hsfe
import glob
from numpy import dot
from numpy.linalg import norm

# HandShapeFeatureExtractor instance
model = hsfe.get_instance()

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
train_gesture_dir = "traindata"
print("train_gesture_dir=",train_gesture_dir)
train_frames_dir = "TrainFrames"
print("train_frames_dir "+train_frames_dir)

train_files = glob.glob('traindata/*.mp4')
# print("train_files=",train_files)
train_frames_path = os.path.join(os.getcwd(),train_frames_dir)
# print("train_frames_path=",train_frames_path)"""""'';"

# Extract the frames for each training video
count = 0
for trainVideoPath in train_files:
    fe.frameExtractor(trainVideoPath, train_frames_path, count)
    count = count + 1

frames_path = os.path.join(train_frames_path,"*.png")
print("frames_path=",frames_path)
frames = glob.glob(frames_path)
print("frames=",frames)

# Extract the feature vector for each training frame
train_feature_list = []
for f in frames:
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feature = model.extract_feature(img)
    feature = np.squeeze(feature)
    train_feature_list.append(feature)
print("train_feature_list=",train_feature_list)


# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
test_gesture_direct = "test"
print("test_gesture_direct=",test_gesture_direct)
test_frames_dir = "TestFrames"
print("test_frames_dir=",test_frames_dir)
print('CWD: ', os.getcwd())

testFiles = glob.glob('test/*.mp4')
print("testFiles=",testFiles)
test_frames_path = os.path.join(os.getcwd(),test_frames_dir)
print("test_frames_path=",test_frames_path)

# Extract the frames for each test video
count = 0
for testVideoPath in testFiles:
    fe.frameExtractor(testVideoPath, test_frames_path, count)
    count = count + 1


frames_path = os.path.join(test_frames_path,"*.png")
print("frames_path=",frames_path)
frames = glob.glob(frames_path)
print("frames=",frames)

# Extract the feature vector for each test frame
test_feature_list = []
for f in frames:
    img = cv2.imread(f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feature = model.extract_feature(img)
    feature = np.squeeze(feature)
    test_feature_list.append(feature)
print("test_feature_list=",test_feature_list)

# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

results = []
for test in test_feature_list:
    cosList = []
    for train in train_feature_list:
        # Compare using cosine similarity
        cos_sim = dot(test, train)/(norm(test)*norm(train))
        cosList.append(cos_sim)
    
    idx = cosList.index(max(cosList))
    results.append(int(idx)%17)
print("results=",results)
    
# Save the list of output indices in 'Results.csv'  
np.savetxt("Results.csv", results, delimiter=',',fmt='%d')
