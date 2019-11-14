
from skimage.measure import compare_ssim
import argparse
import imutils
import cv2




argpfoto = argparse.ArgumentParser()
argpfoto.add_argument("-f", "--first", required=True,
	help="first input image")
argpfoto.add_argument("-s", "--second", required=True,
	help="second")
args = vars(argpfoto.parse_args())





ilkfoto = cv2.imread(args["first"])
ikincifoto = cv2.imread(args["second"])




# dosyalari grayscale dondur
gray_ilk = cv2.cvtColor(ilkfoto, cv2.COLOR_BGR2GRAY)
gray_iki = cv2.cvtColor(ikincifoto, cv2.COLOR_BGR2GRAY)





(benzerlikorani, diff) = compare_ssim(gray_ilk, gray_iki, full=True)
diff = (diff * 255).astype("uint8")
print("iki fotografin benzerlik oranlari: {}".format(benzerlikorani))




cikarma = cv2.threshold(diff, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(cikarma.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)



for c in cnts:
                 #konturlerde loop ikle farkliliklari bulup kutu acar
	# cnts farkliliklari kirmizi ile cizmek icin 3 border kalinlik
	(x, y, w, h) = cv2.boundingRect(c)
	cv2.rectangle(ilkfoto, (x, y), (x + w, y + h), (255, 200, 0), 3)
	cv2.rectangle(ikincifoto, (x, y), (x + w, y + h), (250, 185, 0), 3)


print("Berk_Ozdoruk")




#cikti alma kisminı ctrl s ile save atabilirsiniz.

cv2.imshow("ilk_foto", ilkfoto)

cv2.imshow("ikinci_foto", ikincifoto)

cv2.imshow("cikarma", diff)

cv2.imshow("foto_farklari", cikarma)

cv2.waitKey(0)
#program calisma suresi koymadim, 0 milisaniye cinsinden arttirilabilir.
