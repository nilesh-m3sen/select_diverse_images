import numpy as np
import cv2
from torchvision.io import read_image

def softmax2(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    y1 = y[:,[-1]]
    a2 = a[:,:-1]
    
    exp_a2 = np.exp(a2)
    sum_exp_a2 = np.sum(exp_a2)
    y2 = exp_a2 / sum_exp_a2
    yy = np.concatenate((y2,y1),axis=1)
    return yy

def plot_label_image(img, annots,CLASS):
    colors = ([0,100,0],[0,0,255],[0,215,255],[0,128,128],[0,0,139],[30,105,210])
    for anno in annots:
        cv2.rectangle(img, (int(anno[0]),int(anno[1])), (int(anno[2]),int(anno[3])), color=colors[int(anno[-1])], thickness=1)
        cv2.putText(img, f"{CLASS[int(anno[-1])]}", (int(anno[0])+5,int(anno[1])+18), 1, 1, color=colors[int(anno[-1])], thickness=2)

def new_letterbox(old_img, new_shape=320):
    img = np.array(old_img)
    h,w = img.shape[:2]
    #color = (127,127,127)
    color = (114,114,114)
    if h>new_shape or w>new_shape:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_CUBIC
    aspect = w/h
    if aspect > 1:
        new_w = new_shape
        new_h = np.round(new_w / aspect).astype(int)

        pad_vert = (new_shape - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:
        new_h = new_shape
        new_w = np.round(new_h * aspect).astype(int)

        pad_horz = (new_shape - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:
        new_h, new_w = new_shape, new_shape
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0


    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=color)

    return scaled_img, (pad_top, pad_bot, pad_left, pad_right), (new_w, new_h)

def preprocess_yolo_dynamic(imglist, half=False, h=320, w=320):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """
    ordered_img = np.empty((1,3,320,320),dtype=np.float32)
    one_shapes,ratios = [],[]

    for img in imglist:
        new_img = img.copy()
        if img.shape[2] == 3:
            new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif len(img.shape) == 2:
            new_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        one_shape = new_img.shape[:2]
        one_shapes.append(one_shape)
        # make resized letterbox
        resized, padding, rat = new_letterbox(new_img)
        if resized.ndim == 2:
            resized = resized[:, :, np.newaxis]

        typed = resized.astype(np.float32)

        scaled = typed / 255.0
        # Swap to CHW
        ordered = np.transpose(scaled, (2, 0, 1))
        
        if padding[0]!=0 and padding[2]==0: # horizontal image
            ratio = (img.shape[1]/w, img.shape[0]/rat[1], (0, padding[0]))
        elif padding[0]==0 and padding[2]!=0: # portrait image
            ratio = (img.shape[1]/rat[0], img.shape[0]/h, (padding[2], 0))
        elif padding[0]==0 and padding[2]==0: # square image
            ratio = (img.shape[1]/w, img.shape[0]/h, (0,0))
        ratios.append(ratio)

        ordered_img = np.append(ordered_img, np.expand_dims(ordered, axis=0), axis=0)

    ordered_img = np.delete(ordered_img,0,axis=0)
    if half:
        ordered_img = ordered_img.astype(np.float16)

    return (ordered_img, one_shapes, ratios) 

def preprocess_yolo(img, half=False, h=320, w=320):
    """
    Pre-process an image to meet the size, type and format
    requirements specified by the parameters.
    """

    new_img = img.copy()
    if img.shape[2] == 3:
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif len(img.shape) == 2:
        new_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    one_shape = new_img.shape[:2]

    # make resized letterbox
    resized, padding, rat = new_letterbox(new_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]

    typed = resized.astype(np.float32)

    scaled = typed / 255.0

    # Swap to CHW
    ordered = np.transpose(scaled, (2, 0, 1))
    
    if padding[0]!=0 and padding[2]==0: # horizontal image
        ratio = (img.shape[1]/w, img.shape[0]/rat[1], (0, padding[0]))
    elif padding[0]==0 and padding[2]!=0: # portrait image
        ratio = (img.shape[1]/rat[0], img.shape[0]/h, (padding[2], 0))
    elif padding[0]==0 and padding[2]==0: # square image
        ratio = (img.shape[1]/w, img.shape[0]/h, (0,0))

    if half:
        ordered = ordered.astype(np.float16)

    return (ordered, one_shape, ratio) 

def preprocess_resnet(img_path,img_size,normalize,depth_path,half=False):
    input_image = read_image(img_path)
    img_tensor = normalize(input_image).numpy()
    img_array = np.fromfile(depth_path, np.uint16)
    raw_depth_image = img_array.reshape(480, 848)
    raw_depth_image = cv2.resize(raw_depth_image, img_size)
    raw_depth_image = raw_depth_image.astype(np.float32)/65535
    raw_depth_image = np.expand_dims(raw_depth_image, axis=0)
    #input_arr = np.concatenate((img_tensor.numpy(),raw_depth_image),axis=1)
    input_arr = np.vstack((img_tensor,raw_depth_image))

    return input_arr

def postprocess_yolo_dynamic(results, shapes, ratios):
    """
    Post-process results to show classifications.
    """
    num_dets = results.as_numpy("num_dets")
    bboxes = results.as_numpy("det_boxes")
    scores = results.as_numpy("det_scores")
    classes = results.as_numpy("det_classes")
    cnt = 0
    result = []

    if not len(num_dets) == 0:
        for detcount,bbox, score, clss,shape,ratio in zip(num_dets,bboxes, scores, classes,shapes,ratios):
            s = np.expand_dims(score[:detcount[0]],1)
            c = np.expand_dims(clss[:detcount[0]],1)
            x = np.clip((bbox[:detcount[0],[0,2]]-ratio[2][0])*ratio[0],0,shape[1])
            y = np.clip((bbox[:detcount[0],[1,3]]-ratio[2][1])*ratio[1],0,shape[0])
            ret = np.concatenate((x,y,s,c),axis=1)
            ret[:,[1,2]]=ret[:,[2,1]]
            result.append(ret)
    return result

def postprocess_yolo(results, shape, ratio):
    """
    Post-process results to show classifications.
    """
    num_dets = results.as_numpy("num_dets")[0][0]
    bboxes = results.as_numpy("det_boxes")[0]
    scores = results.as_numpy("det_scores")[0]
    classes = results.as_numpy("det_classes")[0]
    cnt = 0
    result = []
    if not num_dets == 0:
        for bbox, score, clss in zip(bboxes, scores, classes):
            cnt += 1
            #print(clss, infer_class)
            if cnt > num_dets:break
            x1 = np.clip((bbox[0]-ratio[2][0])*ratio[0],0,shape[1])
            y1 = np.clip((bbox[1]-ratio[2][1])*ratio[1],0,shape[0])
            x2 = np.clip((bbox[2]-ratio[2][0])*ratio[0],0,shape[1])
            y2 = np.clip((bbox[3]-ratio[2][1])*ratio[1],0,shape[0])
            result.append([round(x1),round(y1),round(x2),round(y2),score,int(clss)])
    return result

def xyxy2yolo(box,w=424,h=240):
    # Converting coordinates (0-1 normalize)
    x = ((box[0] + box[2]) / 2) / w  # x center
    y = ((box[1] + box[3]) / 2) / h  # y center
    w = (box[2] - box[0]) / w  # width
    h = (box[3] - box[1]) / h  # height
    return (x,y,w,h)

