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

def white_balance2(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result

def white_balance(img):
    max_val = np.max(img)
    blue_channel = img[:,:,0]/max_val
    green_channel = img[:,:,1]/max_val
    red_channel = img[:,:,2]/max_val
    Shape2D = green_channel.shape
    avg_blue = np.average(blue_channel)
    avg_green = np.average(green_channel)
    avg_red = np.average(red_channel)
    result = img.copy()

    if(avg_green-avg_blue > 0.1):
      result[:,:,0] = (blue_channel + ((avg_green - avg_blue) * (np.ones(Shape2D) - blue_channel) * green_channel))*max_val
    result[:,:,2] = (red_channel + ((avg_green - avg_red) * (np.ones(Shape2D) - red_channel) * green_channel))*max_val
    result = white_balance2(result)#.astype("int")
    return result

def adjust_gamma(img, gamma = 0.5):
  if( gamma!=0.5 ):
      curr_table = np.array([((i / 255.0) ** (1.0/gamma)) * 255 for i in np.arange(0, 256)]).astype("uint8")
      return cv2.LUT(img, curr_table) 
  return cv2.LUT(img, table_gamma)

def unsharp_masking(img):
  gaussian_3 = cv2.GaussianBlur(img, (5, 5), 2.0)
  unsharp_image = cv2.addWeighted(img, 2.0, gaussian_3, -1.0, 0)
  return unsharp_image
  #return np.clip((img + 3*(img - gauss_img)), 0,255)

def comp_saliency_old(img):
  #saliency = cv2.saliency.StaticSaliencyFineGrained_create()
  out_img = np.zeros(img.shape)
  (success, saliencyMap) = saliency.computeSaliency(img)
  #saliencyMap = (saliencyMap * 255).astype("uint8")
  for _ in range(img.shape[2]):
    out_img[:,:,_] = saliencyMap/3
  return out_img

saliency = cv2.saliency.StaticSaliencyFineGrained_create()
def comp_saliency2(img):
  #saliency = cv2.saliency.MotionSaliencyBinWangApr2014_create()
  out_img = np.zeros(img.shape)
  (success, saliencyMap) = saliency.computeSaliency(img)
  #saliencyMap = (saliencyMap * 255).astype("uint8")
  #saliencyMap = saliencyMap/np.max(saliencyMap)
  for i in range(3):
    out_img[:,:,i] = saliencyMap
  return out_img/6

bin_kernel = np.array([[1,4,6,4,1],[4,16,24,16,4], [6,24,36,24,6], [4,16,24,16,4],[1,4,6,4,1]])
bin_kerlel = bin_kernel/np.sum(bin_kernel)
def comp_saliency(img):
  lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
  outimg = np.zeros(img.shape)
  dist = np.linalg.norm(-1* cv2.filter2D(lab,-1, bin_kernel) + np.mean(lab, axis = (0,1)), axis = 2)
  dist = dist/np.max(dist)
  for chan in range(3):
    outimg[:,:,chan] = dist
  return outimg

def comp_saturation2(img):
  out_img = np.zeros(img.shape)
  lum = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[...,2]
  #cv2_imshow(img[:,:,0]-lum)
  summ = np.sum(np.array([np.square(img[:,:,elem] - lum) for elem in range(3)]), axis = 0)
  res = np.array(np.sqrt(summ/3))
  for _ in range(img.shape[2]):
    out_img[:,:,_] = res/3
  return out_img

def comp_saturation(img):
  out_img = np.zeros(img.shape)
  lum = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[...,0]
  #cv2_imshow(img[:,:,0]-lum)
  sum = np.sum(np.array([np.square(img[:,:,elem] - lum) for elem in range(3)]), axis = 0)
  res = np.array(np.sqrt(sum))
  res = res/np.max(res)
  for cha in range(img.shape[2]):
    out_img[:,:,cha] = res
  return out_img

def comp_saturation3(img):
  out_img = np.zeros(img.shape)
  lum = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[...,2]
  #cv2_imshow(img[:,:,0]-lum)
  sum = np.sum(np.array([np.square(img[:,:,elem] - lum) for elem in range(3)]), axis = 0)
  res = np.array(np.sqrt(sum))
  res = res/np.max(res)
  for cha in range(img.shape[2]):
    out_img[:,:,cha] = res
  return out_img

def gauss_pyramide(img, layers = 3):
  gaussian_layer = img.copy()
  gaussian = [gaussian_layer]
  for i in range(layers+1):
    gaussian_layer = cv2.pyrDown(gaussian_layer)
    gaussian.append(gaussian_layer)
    #cv2_imshow(gaussian_layer)
  return gaussian

def lap_pyramide2(img, gaussian):
  laplacian = []
  all_ = gaussian
  for i in range(0, len(all_)-1):
    csize = (all_[i].shape[1], all_[i].shape[0])
    upsampled_gaussian = cv2.pyrUp(all_[i+1], dstsize=csize)
    laplacian.append(cv2.subtract(all_[i], upsampled_gaussian))
  #laplacian.append(gaussian[-1])
  return laplacian

def lap_pyramide(img, gaussian):
  laplacian = []
  #all_ = gaussian
  for i in range(0, len(gaussian)-1):
    csize = (gaussian[i].shape[1], gaussian[i].shape[0])
    #print(all_[i].shape,all_[i+1].shape)
    upsampled_gaussian = cv2.pyrUp(gaussian[i+1], dstsize=csize)
    laplacian.append(cv2.subtract(gaussian[i], upsampled_gaussian))
  laplacian.append(gaussian[-1] * 0.15)
  return laplacian


def normalize_maps(*args):
  theta = 0.001
  sum_all = [] 
  for elem in args:
    sum_all.append(sum(elem)) #  + 0.1*np.ones(elem[0].shape)
  norm_all = []
  norm_val = np.sum(np.array([elem + (2*theta*np.ones(elem[0].shape)) for elem in sum_all]), axis = 0)
  for elem in sum_all:
    norm_all.append((elem)/norm_val)
  return norm_all


def naive_fuision(wbImgs, weightMaps):
  sum = []
  for wb, wm in zip(wbImgs, weightMaps):
    sum.append(wm * wb)
  return np.sum(np.array(sum), axis = 0)


bin_kernel = np.array([[1,4,6,4,1],[4,16,24,16,4], [6,24,36,24,6], [4,16,24,16,4],[1,4,6,4,1]])
bin_kerlel = bin_kernel/np.sum(bin_kernel)
def contrast_regional(img):
  luminance_channel = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:,:,0]
  luminance_channel_smoothed =  cv2.filter2D(luminance_channel, -1, bin_kernel)
  diff = luminance_channel - luminance_channel_smoothed
  result = np.zeros(img.shape)
  for elem in range(3):
    result[:,:,elem] = diff
  return result

def exposednes(img):
  c_img = img/np.max(img)
  lambd = 0.25
  Shape3D = img.shape
  #return np.square(img - np.ones(Shape3D)*0.5)
  return np.exp(np.square(c_img - np.ones(Shape3D)*0.5)/(-2*(lambd**2))) * np.max(img)

def pyramide_fusion2(wbImgs, weightMaps, layers = 2, dynamic = 2): #wb is orignal images wm are weight maps calculated by the different algorithms
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

def pyramide_fusion(wbImgs, weightMaps, layers = 2, dynamic = 3): #wb is orignal images wm are weight maps calculated by the different algorithms
  all_out = []
  upsample_size = wbImgs[0].shape
  end_img = np.zeros(upsample_size)
  k_list = [] 
  for wb, wm in zip(wbImgs, weightMaps):
    curr_out = []
    g_wb = gauss_pyramide(histogram_normalization(wb).astype(np.uint8), layers = layers)
    g_wm = gauss_pyramide(wm, layers = layers)

    l_wb = lap_pyramide2(wb, g_wb)
    outimg = np.zeros(upsample_size)
    curr_k = []
    for g, l in zip(g_wm, l_wb):
      outimg = (g+0.5)*l
      curr_k.append(outimg)
    
    curr_k.append(g_wb[-1]*0.15)
    k_list.append(curr_k) 
  
  for a,b in zip(k_list[0], k_list[1]):
    all_out.append(cv2.resize(a+b, (upsample_size[1], upsample_size[0]), interpolation = cv2.INTER_CUBIC))

  final_out = sum(all_out)
  return np.clip(final_out, 0, 255)


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

def mylaplacian(img):
    labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    labimg[:,:,0] = np.abs(cv2.Laplacian(labimg[:,:,0], 8))
    outimg = cv2.cvtColor(labimg, cv2.COLOR_LAB2BGR)
    return outimg/np.max(outimg)

class FUSION():
  def __init__(self, layers = 8, dynamic = 2):
    self.layers = layers 
    self.dynamic = dynamic
  
  def __call__(self, image: np.ndarray, bboxes = None, labels = None):
    imgwb = white_balance(image.copy())

    imgwb_gamma = adjust_gamma(imgwb)
    imgwb_border = unsharp_masking(imgwb)

    L1 = mylaplacian(imgwb_gamma)
    L2 = mylaplacian(imgwb_border)

    #reg_contr1 = contrast_regional(imgwb_gamma)
    #reg_contr2 = contrast_regional(imgwb_border)

    SAL1 = comp_saliency(imgwb_gamma)
    SAL2 = comp_saliency(imgwb_border)

    SAT1 = comp_saturation(imgwb_gamma)
    SAT2 = comp_saturation(imgwb_border)

    NORM1, NORM2 = normalize_maps([L1, SAL1, SAT1], [L2, SAL2, SAT2])
    return np.clip(pyramide_fusion([imgwb_gamma, imgwb_border], [NORM1, NORM2], layers = self.layers, dynamic = self.dynamic), 0, 255).astype(int), bboxes, labels
