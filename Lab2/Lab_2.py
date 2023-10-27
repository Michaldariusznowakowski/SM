import numpy as np
import matplotlib.pyplot as plt
import cv2
import functions.docxSave  as ds
fig = plt.figure()
ax = fig.add_subplot(111)

document = ds.DocxSave()
document.addHeading("Część 1",1)
document.addParagraph("Część pierwsza — wczytywanie obrazów.")
document.addParagraph("plt.imread('Lab2/data/A1.png')")
document.addParagraph("lub cv2.imread('nazwa_pliku.roz')")
img2 = cv2.imread('Lab2/data/A3.png')

img = plt.imread('Lab2/data/A1.png')
document.addParagraph("img.dtype")
document.addParagraph(str(img.dtype))
document.addParagraph("img.shape")
document.addParagraph(str(img.shape))
document.addParagraph("np.min(img),np.max(img)")
document.addParagraph(str(np.min(img))+" "+str(np.max(img)))
ax.imshow(img)
document.addImage(fig,1)


document.addParagraph("Zadanie Obraz 1")


def imgToUInt8(img):
    return img/255.0

def imgToFloat(img):
    return (img*255).astype("uint8")


document.addParagraph("np.issubdtype(img.dtype,np.integer)")
document.addParagraph(np.issubdtype(img.dtype,np.integer))
document.addParagraph("np.issubdtype(imgToUInt8(img).dtype,np.unsignedinteger)")
document.addParagraph(np.issubdtype(imgToUInt8(img).dtype,np.unsignedinteger))
document.addParagraph("np.issubdtype(imgToFloat(img).dtype,np.floating)")
document.addParagraph(np.issubdtype(imgToFloat(img).dtype,np.floating))
document.addHeading("Część druga — wyświetlanie obrazów",1)
document.addParagraph("plt.imshow(img)")
img = plt.imread('Lab2/data/A3.png')
# plt.imshow(img)
ax.imshow(img)
document.addImage(fig,1)




document.addParagraph("Wyświetlcie na ekranie jedną warstwę koloru — dowolną czerwoną, zielona lub niebieską (przykład czerwony).")
R=img[:,:,0]
ax.imshow(R)
document.addImage(fig,1)

document.addParagraph("Jakie wartości powinny przyjąć w zależności od naszych typów danych? maksymalne i minimalne z obrazka np.min(image i np.max(image))")
vminV = np.min(R)
vmaxV = np.max(R)

ax.imshow(R, cmap=plt.cm.gray, vmin=vminV, vmax=vmaxV)
document.addImage(fig,1)

document.addParagraph("Teraz przechodzimy do dokładniejszej metody zamiany obrazu kolorowego na obraz w skali odcieni szarości.")
document.addParagraph("img = np.dot(img[...,:3], [0.2126, 0.7152, 0.0722])")
gray_image = np.dot(img[...,:3], [0.2126, 0.7152, 0.0722])
ax.imshow(gray_image, cmap='gray')
document.addImage(fig,1)
document.addParagraph("Sprawdźmy teraz jak wygląda wyświetlony obraz wczytany za pomocą OpenCV, w ten sam sposób jak na początku.")
R=img2[:,:,0]
gray_image = np.dot(img[...,:3], [0.2126, 0.7152, 0.0722])
ax.imshow(gray_image, cmap='gray')
document.addParagraph("Teraz naprawmy błąd. zamienimy BGR na RGB")
img_RGB = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_BGR = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2BGR)
R=img_BGR[:,:,0]
gray_image = np.dot(img[...,:3], [0.2126, 0.7152, 0.0722])
ax.imshow(gray_image, cmap='gray')
document.addImage(fig,1)
document.addParagraph("Open cv wczytuje obrazy w innym formacie, format BGR, z tego powodu kolory były zamienione")

document.addHeading("Część trzecia — wstęp przejścia do przechodzenia pomiędzy typami danych",1)

img = plt.imread("Lab2/data/lena.png")

gray_image1 = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
gray_image2 = np.dot(img[..., :3], [0.2126, 0.7152, 0.0722])

red_image = img.copy()
red_image[:, :, 1:] = 0 
green_image = img.copy()
green_image[:, :, [0, 2]] = 0
blue_image = img.copy()
blue_image[:, :, :2] = 0



fig1, axs = plt.subplots(3, 3, figsize=(10, 10))
axs[0,0].imshow(img)
axs[0,0].set_title('O - Oryginalny')

axs[0,1].imshow(gray_image1, cmap='gray')
axs[0,1].set_title('Y1 - Odcienie szarości (wzór 1)')

axs[0,2].imshow(gray_image2, cmap='gray')
axs[0,2].set_title('Y2 - Odcienie szarości (wzór 2)')

for i, channel in enumerate(['R', 'G', 'B']):
    axs[1,i].imshow(img[..., i], cmap='gray')
    axs[1,i].set_title(f'{channel} - Warstwa')

axs[2,0].imshow(red_image)
axs[2,0].set_title('Red Image')

axs[2,1].imshow(green_image)
axs[2,1].set_title('Green Image')

axs[2,2].imshow(blue_image)
axs[2,2].set_title('Blue Image')

document.addImage(fig1,4)

document.addParagraph("Zadanie Obraz 3")
img = plt.imread("Lab2/data/B02.jpg")

data = [[0,0,200,200],[200,200,400,400],[400,400,600,600]]
for d in data:
    fragment = img[d[0]:d[2],d[1]:d[3]].copy()
    gray_image1 = np.dot(fragment[..., :3], [0.2989, 0.5870, 0.1140])
    gray_image2 = np.dot(fragment[..., :3], [0.2126, 0.7152, 0.0722])
    red_image = fragment.copy()
    red_image[:, :, 1:] = 0 
    green_image = fragment.copy()
    green_image[:, :, [0, 2]] = 0
    blue_image = fragment.copy()
    blue_image[:, :, :2] = 0
    fig1, axs = plt.subplots(3, 3, figsize=(10, 10))
    axs[0,0].imshow(fragment)
    axs[0,0].set_title('O - Oryginalny')

    axs[0,1].imshow(gray_image1, cmap='gray')
    axs[0,1].set_title('Y1 - Odcienie szarości (wzór 1)')

    axs[0,2].imshow(gray_image2, cmap='gray')
    axs[0,2].set_title('Y2 - Odcienie szarości (wzór 2)')

    for i, channel in enumerate(['R', 'G', 'B']):
        axs[1,i].imshow(fragment[..., i], cmap='gray')
        axs[1,i].set_title(f'{channel} - Warstwa')

    axs[2,0].imshow(red_image)
    axs[2,0].set_title('Red Image')

    axs[2,1].imshow(green_image)
    axs[2,1].set_title('Green Image')

    axs[2,2].imshow(blue_image)
    axs[2,2].set_title('Blue Image')

    document.addImage(fig1,4)
document.save("Lab2.docx")

