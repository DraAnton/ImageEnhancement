import cv2
import numpy as np

GAMMA = 0.6
table_gamma = np.array([((i / 255.0) ** (1.0/GAMMA)) * 255
		for i in np.arange(0, 256)]).astype("uint8")

#saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
saliency = cv2.saliency.StaticSaliencyFineGrained_create()


def histogram_normalization(img):
  img_shape = img.shape
  out_img = np.zeros(img_shape)
  for channel in range(img_shape[2]):
    dyn = 3
    #min_ = np.mean(img[:,:,channel]) - np.std(img[:,:,channel]) * dyn
    #max_ = np.mean(img[:,:,channel]) + np.std(img[:,:,channel]) * dyn
    min_ = np.min(img[:,:,channel])
    max_ = np.max(img[:,:,channel])
    out_img[:,:,channel] = np.clip(255*(img[:,:,channel]-min_*np.ones(img_shape[0:2]))/(max_-min_), 0, 255)
  return out_img.astype(int)

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def adjust_gamma(img, gamma = 0.5):
  if( gamma!=0.5 ):
      curr_table = np.array([((i / 255.0) ** (1.0/gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
      return cv2.LUT(img, curr_table) 
  return cv2.LUT(img, table_gamma)

def unsharp_masking(img):
  gauss_img = cv2.GaussianBlur(img, (5,5), sigmaX = 9, sigmaY = 9)
  out = img*0.95 + histogram_normalization(3*(img - gauss_img))*0.05
  return out.astype("uint8")
  #return np.clip((img + 3*(img - gauss_img)), 0,255)

def comp_saliency(img):
  #saliency = cv2.saliency.StaticSaliencyFineGrained_create()
  out_img = np.zeros(img.shape)
  (success, saliencyMap) = saliency.computeSaliency(img)
  saliencyMap = (saliencyMap * 255).astype("uint8")
  for _ in range(img.shape[2]):
    out_img[:,:,_] = saliencyMap
  return out_img

def comp_saturation(img):
  out_img = np.zeros(img.shape)
  lum = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[...,2]
  #cv2_imshow(img[:,:,0]-lum)
  summ = np.sum(np.array([np.square(img[:,:,elem] - lum) for elem in range(3)]), axis = 0)
  res = np.array(np.sqrt(summ/3))
  for _ in range(img.shape[2]):
    out_img[:,:,_] = res

  return out_img

def gauss_pyramide(img, layers = 3):
  gaussian_layer = img.copy()
  gaussian = [gaussian_layer]
  for i in range(layers+1):
    gaussian_layer = cv2.pyrDown(gaussian_layer)
    gaussian.append(gaussian_layer)
    #cv2_imshow(gaussian_layer)
  return gaussian

def lap_pyramide(img, gaussian):
  laplacian = []
  for i in range(len(gaussian)-1,0,-1):
    size = (gaussian[i - 1].shape[1], gaussian[i - 1].shape[0])
    gaussian_expanded = cv2.pyrUp(gaussian[i], dstsize=size)
    laplacian_layer = cv2.subtract(gaussian[i-1], gaussian_expanded)
    laplacian.append(laplacian_layer)
    #cv2_imshow(laplacian_layer)
  return laplacian


def normalize_maps(*args):
  sum_all = [] 
  for elem in args:
    sum_all.append(sum(elem) + 0.1*np.ones(elem[0].shape))
  norm_all = []
  norm_val = np.sum(np.array(sum_all), axis = 0)
  for elem in sum_all:
    norm_all.append(elem/norm_val)
  return norm_all

def naive_fuision(wbImgs, weightMaps):
  sum = []
  for wb, wm in zip(wbImgs, weightMaps):
    sum.append(wm * wb)
  return np.sum(np.array(sum), axis = 0)

def pyramide_fusion(wbImgs, weightMaps, layers = 2, dynamic = 2): #wb is orignal images wm are weight maps calculated by the different algorithms
  all_out = []
  upsample_size = wbImgs[0].shape
  for wb, wm in zip(wbImgs, weightMaps):
    curr_out = []
    g_wb = gauss_pyramide(wb, layers = layers)
    g_wm = gauss_pyramide(wm, layers = layers)

    l_wb = lap_pyramide(wb, g_wb)
    this_iteration = {"G_PYRAMID":g_wm, "L_PYRAMID":list(reversed(l_wb))}
    all_out.append(this_iteration)  
  
  outimg_list = []
  for l in range(layers):
    g_layers = [elem["G_PYRAMID"][l] for elem in all_out]
    l_layers = [elem["L_PYRAMID"][l] for elem in all_out]

    current_layer = sum([elem_g*elem_l for elem_g, elem_l in zip(g_layers, l_layers)])
    outimg_list.append(cv2.resize(current_layer, (upsample_size[1], upsample_size[0]), interpolation = cv2.INTER_CUBIC))
  
  final_out = sum(outimg_list)
  #for channel in range(3):
  #  curr_img = final_out[:,:,channel]

  #  min = np.mean(curr_img) - np.std(curr_img) * dynamic
  #  max = np.mean(curr_img) + np.std(curr_img) * dynamic
  #  final_out[:,:,channel] = np.clip((curr_img-np.ones(curr_img.shape)*min)/(max-min) * 255, 0, 255)
  
  #return final_out
  min = np.mean(final_out) - np.std(final_out) * dynamic
  max = np.mean(final_out) + np.std(final_out) * dynamic

  return np.clip((final_out-np.ones(final_out.shape)*min)/(max-min) * 230, 0, 255)

def enhance_outdated(image : np.ndarray, layers : int = 8) -> np.ndarray:

    imgwb = white_balance(image.copy())

    imgwb_gamma = adjust_gamma(imgwb)
    imgwb_border = unsharp_masking(imgwb)

    L1 = cv2.Laplacian(imgwb_gamma, 8)
    L2 = cv2.Laplacian(imgwb_border, 8)

    SAL1 = comp_saliency(imgwb_gamma)
    SAL2 = comp_saliency(imgwb_border)

    SAT1 = comp_saturation(imgwb_gamma)
    SAT2 = comp_saturation(imgwb_border)

    NORM1, NORM2 = normalize_maps([L1, SAL1, SAT1], [L2, SAL2, SAT2])
    return np.clip(pyramide_fusion([imgwb_gamma, imgwb_border], [NORM1, NORM2], layers = layers), 0, 255).astype(int)

class FUSION():
  def __init__(self, layers = 8, dynamic = 2):
    self.layers = layers 
    self.dynamic = dynamic
  
  def __call__(self, image: np.ndarray, bboxes = None, labels = None):
    imgwb = white_balance(image.copy())

    imgwb_gamma = adjust_gamma(imgwb)
    imgwb_border = unsharp_masking(imgwb)

    L1 = cv2.Laplacian(imgwb_gamma, 8)
    L2 = cv2.Laplacian(imgwb_border, 8)

    SAL1 = comp_saliency(imgwb_gamma)
    SAL2 = comp_saliency(imgwb_border)

    SAT1 = comp_saturation(imgwb_gamma)
    SAT2 = comp_saturation(imgwb_border)

    NORM1, NORM2 = normalize_maps([L1, SAL1, SAT1], [L2, SAL2, SAT2])
    return np.clip(pyramide_fusion([imgwb_gamma, imgwb_border], [NORM1, NORM2], layers = self.layers, dynamic = self.dynamic), 0, 255).astype(int), bboxes, labels
