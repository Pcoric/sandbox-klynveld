import os
import http.client, urllib.request, urllib.parse, urllib.error, base64
import requests, time, json
import numpy as np

# Maybe not necessary, since i'm not passing arguments (yet)
'''
parser = argparse.ArgumentParser(description="Start a tensorflow model serving")

parser.add_argument('--input', dest="blob_input_data", required=True)
parser.add_argument('--output', dest="output_data1", required=True)

args = parser.parse_args()
'''
def init():
    step_main = 's200_sheet_classifier'
    print('Running step: {}'.format(step_main))

def run(mini_batch):
    print(f'run method start: {__file__}, run({mini_batch})')
    print(mini_batch)
    headers = {
    # Request headers
    'Content-Type': 'application/octet-stream',
    'Prediction-Key': '7129945c0e714b428e60cef9f90c56ad',
    }
    
    endpoint_url = "https://eastus2.api.cognitive.microsoft.com/customvision/v3.0/Prediction/dcc66f32-f086-4609-b099-d71743ee97a7/classify/iterations/Iteration12/image"
    resultList = []

    step_name = "sheet_classifier"

    for file in mini_batch:
        #fullfile = os.getcwd() + "\\test images\\" + file
        basename = os.path.basename(file)
        print("File: " + file)
        #data = open(fullfile,"rb").read()
        data = open(".\\test images\\" + file,"rb").read()
        response = requests.post(endpoint_url, headers=headers, data=data)
        
        root_dict = {}
        root_dict = response.json()
        root_dict["name"] = basename
        root_dict["step_name"] = step_name
        
        if response.status_code != 200:
            root_dict['error'] = {}
            root_dict['error']['status_code'] =  response.status_code
            best_result = 'error'
            #best_result = response.status_code
            #print("request failed for file: ", file)
            #print("response.json", response.json)
        else:
            #print (response.json)
            predictions = response.json()["predictions"]
            prob = [x["probability"] for x in predictions]
            prob_class = [x["tagName"] for x in predictions]
            best_result = prob_class[np.argmax(prob)]

        root_dict['result'] = {}
        root_dict['result']['sheet_type'] = best_result
            
        resultList.append(json.dumps(root_dict))

    return resultList


if __name__ == "__main__":
    # Test scoring
    init()
    # Need to replace with image data, but it's been passed. (for local we'll call out to the images folder, and )
    #test_images = os.path.join(args.input,"")
    print("OS Path: {}".format(os.path))
    filedirectory = os.getcwd()
    print("File Directory: {}".format(filedirectory))
    fullpat = filedirectory + "\\test images\\"
    test_images = os.listdir(fullpat)
    print("test images: {}".format(test_images))
    prediction = run(fullpat)
    # Are we passing this a folder or multiple images???
    print("Test result: ", prediction)
