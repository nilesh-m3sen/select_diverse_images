from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import cv2
import numpy as np
from util import  preprocess_yolo,postprocess_yolo
import time
import os
            
import grpc
from tritonclient.grpc import service_pb2, service_pb2_grpc
from torchvision import transforms
from torchvision.transforms import v2 as T
import torch
from tqdm import tqdm
yolo_ver1_class = {
    0:"Stand_1",
    1:"Sleeping",
    2:"Empty",
    3:"Stand_2",
    4:"Lying",
    5:"Sitting"
}
yolo_ver3_class = {
    0:"Stand_Positive",
    1:"Sleeping",
    2:"Empty",
    3:"Stand_Negative",
    4:"Lying",
    5:"Sitting"
}
#def classify_images():
def make_save_path(save_dir):
    for clsnum,val in yolo_ver1_class.items():
        base_path = os.path.join(save_dir,f"{clsnum}")
        os.makedirs(base_path,exist_ok=True)
        img_save_path = os.path.join(base_path,f"{clsnum}_image")
        os.makedirs(img_save_path,exist_ok=True)
        label_save_path = os.path.join(base_path,f"{clsnum}_label")
        os.makedirs(label_save_path,exist_ok=True)
        pred_save_path = os.path.join(base_path,f"{clsnum}_predict")
        os.makedirs(pred_save_path,exist_ok=True)
        bin_save_path = os.path.join(base_path,f"{clsnum}_bin")
        os.makedirs(bin_save_path,exist_ok=True)
        with open(f'{label_save_path}/classes.txt','w') as f:
            [f.write(x+"\n") for x in yolo_ver1_class.values()]
        if clsnum in [0,3]:
            ver3label_save_path = os.path.join(base_path,f"{clsnum}_ver3_label")
            os.makedirs(ver3label_save_path,exist_ok=True)
            ver3img_save_path = os.path.join(base_path,f"{clsnum}_ver3_image")
            os.makedirs(ver3img_save_path,exist_ok=True)
            with open(f'{ver3label_save_path}/classes.txt','w') as f:
                [f.write(x+"\n") for x in yolo_ver3_class.values()]

def move_data(infer_model_name,grpc_url,img_path,img_list,save_path,bin_path=None,half=False):
    import shutil
    save_count=1
    with grpcclient.InferenceServerClient(url=grpc_url, verbose=False) as client:
        for no, imgs in enumerate(tqdm(img_list),start=1):
            filename = os.path.splitext(os.path.basename(imgs))[0]
            
            if bin_path is not None:
                bin_file = f"{filename}.bin"
                bin_data_path = os.path.join(bin_path,bin_file)
                if not os.path.isfile(bin_data_path):
                    print(f"Not Exist : {bin_file}")
                    continue
  
            data_path = os.path.join(img_path,imgs)

            img = cv2.imread(data_path)

            preprocessed_img, one_shape, ratio = preprocess_yolo(img,half)
            # dynamic input

            outputs = [grpcclient.InferRequestedOutput("num_dets"),
                    grpcclient.InferRequestedOutput("det_boxes"),
                    grpcclient.InferRequestedOutput("det_scores"),
                    grpcclient.InferRequestedOutput("det_classes"),
                    ]
            inputs = [grpcclient.InferInput("images", (1,3,320,320), np_to_triton_dtype(np.float16)),]
            inputs[0].set_data_from_numpy(np.expand_dims(preprocessed_img, axis=0))

            start = time.time()
            response = client.infer(model_name = infer_model_name,
                                        inputs = inputs,
                                        outputs = outputs)
            end = time.time()
            print(f"Triton Yolov11 Inference Time {end - start:.5f} sec")
            result = postprocess_yolo(response,one_shape,ratio)
            if len(result)>0:
                maxindex = np.argmax(np.array(result)[:,4])
                pred_class = int(result[maxindex][-1])
                target_path = os.path.join(save_dir,str(pred_class),f"{pred_class}_predict",f"{filename}.jpg")
                shutil.copy(data_path, target_path)
                if bin_path is not None:
                    bin_target_path = os.path.join(save_dir,str(pred_class),f"{pred_class}_bin",f"{filename}.bin")
                    shutil.copy(bin_data_path, bin_target_path)
                    save_count+=1
        print(f"file Count : {no}, save_count : {save_count}")


if __name__=="__main__":
    HALF = True
    grpc_url = "192.168.123.171:8001"
    channel = grpc.insecure_channel(grpc_url)
    grpc_stub = service_pb2_grpc.GRPCInferenceServiceStub(channel)

    # Health Check
    try:
        request = service_pb2.ServerLiveRequest()
        response = grpc_stub.ServerLive(request)
        print("Triton server health check {}".format(response))
    except Exception as ex:
        print(ex)

    infer_model_name = "yolov11_ver1_12882"
    base_path = "E:/jan_13_data/DW/20250111/RGB_selected"
    img_path = f"{base_path}"
    bin_path = None
    img_list = os.listdir(img_path)
    save_dir = f"{base_path}_label"

    make_save_path(save_dir)
    move_data(infer_model_name,grpc_url,img_path,img_list,save_dir,bin_path,half=True)

