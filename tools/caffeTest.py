# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from PIL import Image
import scipy.misc as sci
import time
import caffe
from optparse import OptionParser
import os
import os.path
import glob
import json
import matplotlib.pyplot as plt
import sys
import csv
import codecs
from processCaffeData import getDataClass, writeCaffeImageData
#python2
reload(sys)
sys.setdefaultencoding('utf-8')

def parse_arguments():

    parser = OptionParser()
    parser.description = \
        "This program takes the smartEye test datas"

    parser.add_option("-i", "--input", dest="input_path",
                      metavar="PATH", type="string",
                      help="path to the "
                      "wissen dataset folder")

    parser.add_option("-p", "--post", action="store", dest="video_post",
                      type="string", default="*.avi", help="video post")

    parser.add_option("-o", "--output", dest="output_path",
                      metavar="DIRECTORY", type="string",
                      help="path to a non existing directory"
                      "where the new training dataset will be created")

    (options, args) = parser.parse_args()
    #print (options, args)

    if options.input_path:
        if not os.path.exists(options.input_path):
            parser.error("Could not find the input file")
        else:
            # we normalize the path
            options.input_path = os.path.normpath(options.input_path)
    else:
        parser.error("'input' option is required to run this program")

    return options

def getDirFiles(dataDir, filePost):
    imagePathPattern = os.path.join(dataDir, filePost)
    for filePath in glob.iglob(imagePathPattern):
        yield filePath
    return

def readClassData(classFilePath):
    resultData = {}
    with open(classFilePath, "r") as classFile:
        resultData = json.load(classFile)
    return resultData

def caffeInit(netPath, modelPath, meanFile):
    net = caffe.Net(netPath, modelPath, caffe.TEST)
    caffe.set_device(0)
    caffe.set_mode_gpu()
    mean_blob = caffe.proto.caffe_pb2.BlobProto()
    mean_blob.ParseFromString(open(meanFile, 'rb').read())
    caffeMean = caffe.io.blobproto_to_array(mean_blob)
    return net, caffeMean

def testCaffeModel(inputPath, outputPath, netPath, modelPath, meanFile):
    caffeNet, caffeMean = caffeInit(netPath, modelPath, meanFile)
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    for imagePath in getDirFiles(inputPath, "*.*"):
        image = caffe.io.load_image(imagePath, False)
        if image is not None:
            print(imagePath)
            path, fileNameAndPost = os.path.split(imagePath)
            fileName, post = os.path.splitext(fileNameAndPost)
            transformer = caffe.io.Transformer({'data': caffeNet.blobs['data'].data.shape})
            transformer.set_transpose('data', (2, 0, 1))
            transformer.set_mean('data', caffeMean[0, :, 0, 0])
            transformer.set_raw_scale('data', 255)
            # transformer.set_channel_swap('data', (2,1,0))
            caffeNet.blobs['data'].data[...] = transformer.preprocess('data', image)
            caffeNet.forward()
            prob = caffeNet.blobs['prob'].data[0]
            classIndex = prob.argmax()
            saveImageDir = os.path.join(outputPath, str(classIndex))
            if not os.path.exists(saveImageDir):
                os.makedirs(saveImageDir)
            saveImagePath = os.path.join(saveImageDir, fileNameAndPost)
            os.rename(imagePath, saveImagePath)

def createCaffeModelResult(inputPath, outputPath, netPath, modelPath, meanFilePath, classFilePath, flag):
    caffeNet, caffeMean = caffeInit(netPath, modelPath, meanFilePath)
    if not os.path.isfile(outputPath):
        print("input csv file")
    classData = readClassData(classFilePath)
    #print(classData)
    resultData = []
    for imagePath in getDirFiles(inputPath, "*.*"):
        image = caffe.io.load_image(imagePath, False)
        if image is not None:
            print(imagePath)
            path, fileNameAndPost = os.path.split(imagePath)
            fileName, post = os.path.splitext(fileNameAndPost)
            transformer = caffe.io.Transformer({'data': caffeNet.blobs['data'].data.shape})
            transformer.set_transpose('data', (2, 0, 1))
            transformer.set_mean('data', caffeMean[0, :, 0, 0])
            transformer.set_raw_scale('data', 255)
            # transformer.set_channel_swap('data', (2,1,0))
            caffeNet.blobs['data'].data[...] = transformer.preprocess('data', image)
            caffeNet.forward()
            prob = caffeNet.blobs['prob'].data[0]

            imageResult = ""
            clsData = []
            if flag == 0:
                classIndex = prob.argmax()
                imageResult = "%s" % (classData[str(classIndex)])
            elif flag == 1:
                topFiveIndex = prob.argsort()[::-1][:5]
                #print(topFiveIndex)
                imageResult = "%s%s%s%s%s" % (classData[str(topFiveIndex[0])], classData[str(topFiveIndex[1])],
                            classData[str(topFiveIndex[2])], classData[str(topFiveIndex[3])],
                            classData[str(topFiveIndex[4])])
            elif flag == 2:
                clsData.append(fileNameAndPost)
                for probability in prob:
                    clsData.append("%f" % probability)

            if flag != 2:
                resultData.append((fileNameAndPost, imageResult))
            else:
                resultData.append(clsData)
    
    if flag != 2:
        with codecs.open(outputPath, "w", encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filename", "label"])
            for imageName, imageResult in resultData:
                writer.writerow([imageName, imageResult])
    else:
        with codecs.open(outputPath, "w", encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
	    if len(resultData) > 0:
                clsNumber = len(resultData[0]) - 1
                headData = ["filename"]
                for x in range(clsNumber):
                    headData.append("p%d" % x)
                writer.writerow(headData)
            	for data in resultData:
                    writer.writerow(data)

def createCaffeHardData(inputPath, outputPath, netPath, modelPath, meanFilePath):
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    dataClass = getDataClass(inputPath)
    caffeNet, caffeMean = caffeInit(netPath, modelPath, meanFilePath)
    saveTrainFilePath = os.path.join(outputPath, "hard_train.txt")
    saveTrainFile = open(saveTrainFilePath, "w")
    for classIndex, className in enumerate(dataClass):
        dataClassDir = os.path.join(inputPath, className)
        for imagePath in getDirFiles(dataClassDir, "*.*"):
            image = caffe.io.load_image(imagePath, False)
            if image is not None:
                print(imagePath)
                path, fileNameAndPost = os.path.split(imagePath)
                fileName, post = os.path.splitext(fileNameAndPost)

                transformer = caffe.io.Transformer({'data': caffeNet.blobs['data'].data.shape})
                transformer.set_transpose('data', (2, 0, 1))
                transformer.set_mean('data', caffeMean[0, :, 0, 0])
                transformer.set_raw_scale('data', 255)
                # transformer.set_channel_swap('data', (2,1,0))
                caffeNet.blobs['data'].data[...] = transformer.preprocess('data', image)
                caffeNet.forward()
                prob = caffeNet.blobs['prob'].data[0]
                classIndex = prob.argmax()
                if classIndex != int(className):
                    saveImageDir = os.path.join(outputPath, str(classIndex))
                    if not os.path.exists(saveImageDir):
                        os.makedirs(saveImageDir)
                    saveImagePath = os.path.join(saveImageDir, fileNameAndPost)
                    writeCaffeImageData(imagePath, saveImageDir, classIndex, saveTrainFile, "train")

    saveTrainFile.close()

def averageModels(modelsResult, ratios, classFilePath, flag, outputPath):
    models = []
    classData = readClassData(classFilePath)
    clsNumber = len(classData)
    for csvFile in modelsResult:
        model = pd.read_csv(csvFile)
        models.append(model)
	#print(type(model))
    if len(ratios) == len(models) and clsNumber > 0:
        ensemble = models[0][['filename']].copy()
        headData = []
        for number in range(clsNumber):
            p = "p%d" % number
            headData.append(p)
            ensemble[p] = np.sum([m[p] * r for m, r in zip(models, ratios)], axis=0)
        resultData = []
	fileNameAndPostList = ensemble[['filename']].as_matrix()
	probDatas = ensemble[headData].as_matrix()
	#print(fileNameAndPostList)
        for index, prob in enumerate(probDatas):
            if flag == 0:
                classIndex = prob.argmax()
                imageResult = "%s" % (classData[str(classIndex)])
            elif flag == 1:
                topFiveIndex = prob.argsort()[::-1][:5]
                # print(topFiveIndex)
                imageResult = "%s%s%s%s%s" % (classData[str(topFiveIndex[0])], classData[str(topFiveIndex[1])],
                                              classData[str(topFiveIndex[2])], classData[str(topFiveIndex[3])],
                                              classData[str(topFiveIndex[4])])
            resultData.append((fileNameAndPostList[index][0], imageResult))

        with codecs.open(outputPath, "w", encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["filename", "label"])
            for imageName, imageResult in resultData:
                writer.writerow([imageName, imageResult])
    else:
        print("input models error!")

def main():
    print("start...")
    #options = parse_arguments()
    #testCaffeModel("/home/wfw/lipj/output", "/home/wfw/lipj/TMD/test2", "/home/wfw/lipj/TMD/ResNet_101_deploy.prototxt",
    #               "/home/wfw/lipj/TMD/snapshot/TMD_new__iter_99000.caffemodel", "/home/wfw/lipj/TMD/train_mean.binaryproto")
    #createCaffeModelResult("/home/wfw/lipj/test2", "/home/wfw/lipj/TMD/result0.csv", "/home/wfw/lipj/TMD/deploy_inception-v4.prototxt",
    #                       "/home/wfw/lipj/TMD/models/TMD_inception_v4_1_100000.caffemodel", "/home/wfw/lipj/TMD/train_mean0.binaryproto",
    #                      "/home/wfw/lipj/TMD/class.json", 2)
    modelsResult = ["/home/wfw/lipj/TMD/result/result1.csv", "/home/wfw/lipj/TMD/result/result0.csv", "/home/wfw/lipj/TMD/result/result.csv"]
    ratios = [0.51, 0.3, 0.19]
    averageModels(modelsResult, ratios, "/home/wfw/lipj/TMD/class.json", 1, "/home/wfw/lipj/TMD/final_result.csv")
    #createCaffeHardData("/home/wfw/lipj/TMD/train1", "/home/wfw/lipj/TMD/hard_train", "/home/wfw/lipj/TMD/ResNet_101_deploy.prototxt",
    #                    "/home/wfw/lipj/TMD/models/TMD_new__iter_100000_resnet1.caffemodel", "/home/wfw/lipj/TMD/train_mean.binaryproto")

    print("End of game, have a nice day!")

if __name__ == "__main__":
    main()



