import cv2
import numpy as np 

def mult__(*args):
  out = 1
  for elem in args:
    out = out * elem
  return out


class MSRCR():
  def __init__(self, sigmas: tuple = (30, 60, 120)):
    self.sigmas = sigmas 
    self.grids = {}
  
  def __calc_grids(self, this_shape: tuple ):
    assert len(this_shape) == 2
    [mx, my]=np.meshgrid(range(0, this_shape[1]), range(0, this_shape[0]))
    key_ = str(this_shape[0])+"-"+str(this_shape[1])

    outlist = []
    for elem in self.sigmas:
      g1 = np.exp(-(mx^2+my^2)/(elem*elem))
      Gauss_1 = (g1/np.sum(g1))
      Gauss_1_fourier = np.fft.fft2(Gauss_1, s = Gauss_1.shape)
      outlist.append(Gauss_1_fourier)
    self.grids[key_] = outlist

  def enhance_outdated(image : np.ndarray, sigmas : tuple = (25, 50, 150), gauss_dim : tuple = (9,9), normalization_multiplyer : float = 3) -> np.ndarray:
    result_image = np.zeros(image.shape)
    colour_corr_matrix = np.zeros(image.shape)

    Shape2D = image.shape[0:2]
    Shape3D = image.shape

    colour_corr_prematrix = 1/mult__(*Shape3D) * np.clip(np.sum(image, axis = 2), 1, 256) 

    for channel_index in [0, 1, 2]:
        single_channel = image[:, :, channel_index].copy() 
        log_single_channel = np.log(single_channel + np.ones(Shape2D))
        fourier_single_channel = np.fft.fft2(single_channel)

        intermediate = np.zeros(Shape2D)

        for curr_sigma in sigmas:
            GG = cv2.GaussianBlur(single_channel, gauss_dim, sigmaX=curr_sigma, sigmaY=curr_sigma)
            Gauss_fourier = np.fft.fft2(GG, s = Shape2D)

            between_=np.fft.ifft2(Gauss_fourier * fourier_single_channel).real
            Rr_log = np.log(np.clip(between_, 0.00001, 255)) 
            
            intermediate = intermediate + (log_single_channel-Rr_log)/len(sigmas)
        
        ColorRestauration = np.log(np.divide(np.clip(single_channel, 1, 255), colour_corr_prematrix))
        intermediate = ColorRestauration * np.exp(intermediate)

        minInter = np.mean(intermediate) - np.std(intermediate) * normalization_multiplyer
        maxInter = np.mean(intermediate) + np.std(intermediate) * normalization_multiplyer

        intermediate = np.clip(255*(intermediate-minInter*np.ones(Shape2D))/(maxInter-minInter), 0, 255)
        result_image[:, :, channel_index] = intermediate
    return result_image


  def __call__(self, image, bboxes = None, labels = None):
    #result_image = np.zeros(colour_image.shape)
    result_imageCr = np.zeros(image.shape)
    colour_corr_matrix = np.zeros(image.shape)

    Shape3D = image.shape
    Shape2D = Shape3D[0:2]

    colour_corr_prematrix = np.clip(np.sum(image, axis = 2),1 ,256)
    key_ = str(Shape2D[0])+"-"+str(Shape2D[1])

    precalculated_grids = None 
    while(precalculated_grids is None):
      precalculated_grids = self.grids.get(key_, None)
      if(precalculated_grids is None):
        self.__calc_grids(Shape2D)

    for col in [0,1,2]:
      Image_channel = image[:,:,col].copy()

      log_channel = np.log(Image_channel + np.ones(Shape2D))
      fourier_channel = np.fft.fft2(Image_channel)

      Rr = 0
      for grid in precalculated_grids:
        between1_=np.fft.ifft2(grid * fourier_channel).real 
        Rr += (log_channel-np.log(between1_))/len(precalculated_grids)

      ColorRestauration = np.log(np.divide(np.clip(Image_channel, 1, 255),1/mult(*Shape3D) * colour_corr_prematrix))#/np.log(np.ones(Shape2D)*40)
      RrCr = ColorRestauration * Rr
    
      min1Cr = np.mean(RrCr) - np.std(RrCr) * dynamic
      max1Cr = np.mean(RrCr) + np.std(RrCr) * dynamic

      RrCr = np.clip(255*(RrCr-min1Cr*np.ones(Shape2D))/(max1Cr-min1Cr), 0, 255)

      result_imageCr[:,:,col] = RrCr.astype(int)
    return result_imageCr, bboxes, labels