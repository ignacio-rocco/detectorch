import numpy as np
import cv2
 
def selective_search(pil_image=None,quality='f',size=800):
    # speed-up using multithreads
    cv2.setUseOptimized(True);
    cv2.setNumThreads(4);

    # resize image to limit number of proposals and to bypass a bug in OpenCV with non-square images
    w,h = pil_image.size
    h_factor,w_factor=h/size,w/size
    pil_image=pil_image.resize((size,size))

    im = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)        
 
    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
 
    # set input image on which we will run segmentation
    ss.setBaseImage(im)
 
    # Switch to fast but low recall Selective Search method
    if (quality == 'f'):
        ss.switchToSelectiveSearchFast()
     # Switch to high recall but slow Selective Search method
    elif (quality == 'q'):
        ss.switchToSelectiveSearchQuality()

    # run selective search segmentation on input image
    rects = ss.process()

    # rect is in x,y,w,h format
    # convert to xmin,ymin,xmax,ymax format
    rects = np.vstack((rects[:,0]*w_factor, rects[:,1]*h_factor, (rects[:,0]+rects[:,2])*w_factor, (rects[:,1]+rects[:,3])*h_factor)).transpose()
    
    return rects