
import cv2 
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
import prep as prep

from skimage import feature
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border
from skimage.color import label2rgb
import numpy.ma as ma

#name='./images/20150302065934Lh.png' # unclear
name='./images/20150302164934Ch.png' # clear
#name='./images/20150301130014Th.png' # clear
#name='20150302025534Lh.png'          #cloudy
#name='20150301000234Lh.png'
#name='./images/20150302164934Ch.png'
imx=cv2.imread(name)
img0 = prep.readim2gray(name)
img_smooth=cv2.medianBlur(img0,13) #give a smooth


cloud=prep.cloud_detection(img_smooth) #this give a cloud detection
if cloud ==1:
    corrected,center,Rsun=prep.remove_limbdark(img0)
else:
    print("Please, change a clear image for filament detection.")
    exit()

corrected=prep.intensity_gray(corrected)
#-------------------------------------
ok=1
if ok==1:
#def detect_filament(corrected):
    # give the threshold for the filament dentection
    blur = 2*corrected-cv2.GaussianBlur(corrected,(21,21),0) #remove the opacity of clouds
    threshold=-0.35*np.std(blur)+np.median(blur)        #define the global threshold
    ret3,threshim = cv2.threshold(blur,threshold,255,cv2.THRESH_BINARY_INV) #give a filterd image
    threshim=prep.zero_limb(threshim,center,Rsun)                #remove the limb 
    #---------------------remove all sporadic features------------------------
    remain_regions = []
    # Features as selected by the surface criteria
    # This can discard sunspots and other small round features
    im_label = label(threshim,connectivity =2)
    for pts in regionprops(im_label):
         if pts.area >=20 and pts.axis_major_length>=10:
              e=pts.axis_major_length/pts.axis_minor_length
              if  e>1.2:
                  remain_regions.append(pts)
    gray = threshim.copy()
    mask = np.zeros(np.shape(gray))
    for k in remain_regions:
         #print('area,length,width=',k.area,k.axis_major_length,k.axis_minor_length)
         x_c = np.zeros(len(k.coords))
         y_c = np.zeros(len(k.coords))

         for i in np.arange(len(k.coords)):
             x_c[i] = k.coords[i][0]
             y_c[i] = k.coords[i][1]
             mask[x_c.astype(int),y_c.astype(int)]=True
    binary=mask.copy()
    # canny edge detection
    edges2 = feature.canny(binary, sigma=1)
    # make a fat kernel	
    kernel = np.ones((5,5),np.uint8)
    #Dilation with this kernel
    dilation = morphology.binary_dilation(edges2,kernel)
    # Erosion to recover the original shape
    erosion = morphology.binary_erosion(dilation,kernel)		
    # Now we want to tag the structures that have a surface bigger than a threshold to only keep filaments and discard spots
    image = erosion.copy()
    # apply threshold
    thresh = threshold_otsu(image)
    #print('threshold_otsu=',thresh)
    bw = morphology.closing(image > thresh, morphology.square(3))
    # remove artifacts connected to image border
    # label image regions
    labelim = label(bw,connectivity =1)
    gray=corrected.copy()
    result = ma.masked_where(mask,gray)
    
    def multi_show(imgs,labels):
        fig = plt.figure(figsize=(7,7))
        for i in range(len(imgs)):
            img = imgs[i]
            title=str(i+1)
            #行，列，索引
            plt.subplot(2,2,i+1)
            plt.imshow(img,cmap='gray')
            plt.title(labels[i])
            #plt.colorbar()
            #plt.title(title,fontsize=8)
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.show()
    imgs = [corrected,threshim,binary, result]
    labels =['corrected','global threshold','binary','result']
    multi_show(imgs,labels)

    #return labelim,regions,threshim,bgk,result,new,image
    
import matplotlib.patches as mpatches
fig, ax = plt.subplots(figsize=(10, 6))
ax.imshow(corrected,cmap='gray')
interesting = []
filament_coords=[]
for region in regionprops(labelim, intensity_image=corrected):
    if region.area>=10:
        print('centroid,area,length,width=',region.centroid,region.area_filled,region.axis_major_length,region.axis_minor_length)
        # take regions with large enough areas
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc-5, minr-5), maxc - minc+10, maxr - minr+10,
                                  fill=False, edgecolor='white', linewidth=1)
        ax.add_patch(rect)
        roi = corrected[minr-5:minr+maxr - minr+5,minc-5:minc+maxc - minc+5]
        filament_coords.append(region.coords)
        interesting.append(roi)

ax.set_axis_off()
plt.tight_layout()
plt.show()




#-------------------------------------------------





print("all is done.")
