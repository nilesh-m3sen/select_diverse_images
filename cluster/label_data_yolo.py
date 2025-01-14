from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import cv2
import numpy as np
from util import preprocess_yolo,postprocess_yolo,xyxy2yolo
import time
import os
import grpc
from tritonclient.grpc import service_pb2, service_pb2_grpc
from tqdm import tqdm

def yolo_label(infer_model_name,grpc_url,img_path,img_list,label_savepath,label_class,half=False):
    import shutil
    labeled,noresult=0,0
    with grpcclient.InferenceServerClient(url=grpc_url, verbose=False) as client:
        for no, imgs in enumerate( tqdm(img_list),start=1):
            filename = os.path.splitext(os.path.basename(imgs))[0]
            data_path = os.path.join(img_path,imgs)
            img = cv2.imread(data_path)
            # print(data_path)
            if os.path.isfile(data_path) and img is None:
                # print(f"There is problem in KR encode : {imgs}")
                ff = np.fromfile(data_path, np.uint8)
                img = cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)
                continue
                #imgsize = img.shape[1::-1]
            
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
            #print(f"Triton Yolov11 Inference Time {end - start:.5f} sec")
            
            result = postprocess_yolo(response,one_shape,ratio)
            if len(result)>0:
                maxindex = np.argmax(np.array(result)[:,4])
                bbox = result[maxindex]
                labeled+=1
                yolocoordlist = xyxy2yolo(bbox[:4])
                #print(f"{label_class} {round(yolocoordlist[0],6)} {round(yolocoordlist[1],6)} {round(yolocoordlist[2],6)} {round(yolocoordlist[3],6)}")
                save_txt_path = os.path.join(label_savepath,filename+".txt")
                with open(save_txt_path, "w") as f:
                    f.write(f"{label_class} {round(yolocoordlist[0],6)} {round(yolocoordlist[1],6)} {round(yolocoordlist[2],6)} {round(yolocoordlist[3],6)}")
            else:
                noresult+=1
        print(f"file Count : {no}, labeled_count : {labeled}, No result : {noresult}")
        
if __name__=="__main__":
    ## model predict fp16 
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

    ## inference model name
    infer_model_name = "yolov11_ver1"
    base_path = f"D:/Nilesh/labeling_work_24_12_23/yolodata/SW/20241226/version_1/f"
    labeled_no_array = [0, 1, 3, 4]
    label_ver = 0


    for labeled_no in labeled_no_array: 
        if label_ver==3:
            img_dir = f"{base_path}/{labeled_no}/{labeled_no}_ver3_image/"
            save_dir = f"{base_path}/{labeled_no}/{labeled_no}_ver3_label/"
        else:
            img_dir = f"{base_path}/{labeled_no}/{labeled_no}_image"
            save_dir = f"{base_path}/{labeled_no}/{labeled_no}_label"
        img_list = os.listdir(img_dir)

        yolo_label(infer_model_name,grpc_url,img_dir,img_list,save_dir,labeled_no,HALF)