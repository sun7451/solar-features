import cv2
import numpy as np
from skimage import morphology
import warnings

warnings.filterwarnings("ignore")
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Prep(object):

    @staticmethod
    def cloud_detection(im):
        """ Given an H-alpha image of the sun, this function returns binary values:
        0 for clear image,
        1 for cloudy image."""
        original = im
        original.astype('uint8')
        original = cv2.medianBlur(original, 13)
        # set the binary image
        ret, thresh1 = cv2.threshold(original, 127, 255, cv2.THRESH_BINARY)
        # detect edge and give a ellipse fit
        binary = cv2.Canny(thresh1, 50, 150)
        cnt, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ellipse = cv2.fitEllipse(cnt[0])  # note that ellipse = [ (x, y) , (a, b), angle ]
        # cloudy detection 
        ab = (ellipse[1])
        # logger.info(ab[0],ab[1])
        e = ab[1] / ab[0]
        if 0.98 < e < 1.02:
            r = 1
        else:
            r = 0
        logger.info("This is a cloudy image with E={}".format(e))
        return r

    def remove_limbdark(self, im, center=None, RSUN=None):
        #  pip install polarTransform==1.0.1
        import polarTransform as pT
        from scipy import signal
        import cv2

        radiusSize, angleSize = 1024, 1800
        im = self.removenan(im)
        im2 = im.copy()
        if center is None:
            T = (im.max() - im.min()) * 0.2 + im.min()
            arr = (im > T)
            import scipy.ndimage.morphology as snm
            arr = snm.binary_fill_holes(arr)
            #        im2=(im-T)*arr
            Y, X = np.mgrid[:im.shape[0], :im.shape[1]]
            xc = (X * arr).astype(float).sum() / (arr * 1).sum()
            yc = (Y * arr).astype(float).sum() / (arr * 1).sum()
            center = (xc, yc)
            RSUN = np.sqrt(arr.sum() / np.pi)
        self.disk = np.int8(self.disk(im.shape[0], im.shape[1], RSUN))
        impolar, Ptsetting = pT.convertToPolarImage(im, center, radiusSize=radiusSize, angleSize=angleSize)
        profile = np.median(impolar, axis=0)
        profile = signal.savgol_filter(profile, 11, 3)
        Z = profile.reshape(-1, 1).T.repeat(impolar.shape[0], axis=0)
        Z = Ptsetting.convertToCartesianImage(Z)
        im2 = self.removenan(im / Z) - 1
        im2 = im2 * self.disk
        im = self.removenan(im - Z)
        im = im * self.disk
        # ----------star to clear all the limb---------------
        imside = self.imnorm(im) * 255
        height, width = im.shape
        RSUN = RSUN.astype(int)
        center = np.rint(center)
        center = center.astype(int)
        outside_image = cv2.bitwise_xor(imside, imside)
        circle_img2 = np.zeros((height, width), np.uint8)
        cv2.circle(circle_img2, (center[0], center[1]), RSUN - 1, 1, thickness=-1)
        ins = cv2.bitwise_and(imside, imside, mask=circle_img2)
        # ----------end to clear all the limb---------------
        # return outside_image, inside_image
        # return im, center, RSUN, self.disk,im2,Z
        return ins, center, RSUN  # , im

    # ---------------other useful def-------------
    @staticmethod
    def readim2gray(pngfile):
        import imageio
        Img0 = imageio.imread(pngfile)
        im = cv2.cvtColor(Img0, cv2.COLOR_BGR2GRAY)
        return im

    def imnorm(self, im, mx=0, mi=0):
        """
        Normalize an image
        """
        if mx != 0 and mi != 0:
            pass
        else:
            mi, mx = np.min(im), np.max(im)
        im2 = self.removenan((im - mi) / (mx - mi))

        arr1 = (im2 > 1)
        im2[arr1] = 1
        arr0 = (im2 < 0)
        im2[arr0] = 0
        return im2

    @staticmethod
    def removenan(im, key=0):
        """
        remove NAN and INF in an image
        """
        im2 = np.copy(im)
        arr = np.isnan(im2)
        im2[arr] = key
        arr2 = np.isinf(im2)
        im2[arr2] = key
        return im2

    @staticmethod
    def disk(M, N, r0):
        X, Y = np.meshgrid(np.arange(int(-(N / 2)), int(N / 2)), np.linspace(-int(M / 2), int(M / 2) - 1, M))
        r = np.sqrt((X) ** 2 + (Y) ** 2)
        im = r < r0
        return im

    @staticmethod
    def intensity_gray(img):
        " Compute the pixel-wise intensity of a non-gray image"
        # img must be a gray image
        # img1 = cv2.imread(self.path)
        # img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        temp = [img[j][i] for j in range(img.shape[0]) for i in range(img.shape[1])]
        intensity = np.reshape(temp, [img.shape[0], img.shape[1]])
        return intensity

    @staticmethod
    def zero_limb(img, center, RSUN):
        # ----------star to clear all the limb---------------
        im = img.copy()
        height, width = im.shape
        RSUN = RSUN.astype(int)
        center = np.rint(center)
        center = center.astype(int)
        outside_image = cv2.bitwise_xor(im, im)
        circle_img2 = np.zeros((height, width), np.uint8)
        cv2.circle(circle_img2, (center[0], center[1]), RSUN - 2, 1, thickness=-1)
        ins = cv2.bitwise_and(im, im, mask=circle_img2)
        # ----------end to clear all the limb---------------
        return ins
