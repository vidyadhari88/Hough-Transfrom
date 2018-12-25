# ubit number: 50290085
# ubit name: vc34
# coding: utf-8

# # task 3 - Hough line Transform Implementation

# In[1]:


import cv2
import numpy as np


# In[36]:


#flipping the kernel for convolusion
def kernel_flip(kernel):
    kernel_copy = kernel.copy()
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
              kernel_copy[i,j] = kernel[kernel.shape[0]-i-1,kernel.shape[1]-j-1]
    return kernel_copy


#convolusion function
def conv(image,kernel):
    flipped_kernel = kernel_flip(kernel)
    image_height = image.shape[0]
    image_width = image.shape[1]
    
    kernel_height = flipped_kernel.shape[0]
    kernel_width = flipped_kernel.shape[1]
    
    h= kernel_height//2
    w= kernel_width//2
    conv_image = np.zeros(image.shape)
    
    for i in range(1, image_height-1):
        for j in range(1,image_width-1):
            sum1 = 0
            for m in range(-1,+2):
                for n in range(-1,+2):
                    sum1 += flipped_kernel[m,n]*image[i+m,j+n]
            conv_image[i,j] = sum1
    return conv_image

#hough line implementation algortihm
def HoughTImple(CannyImage,image,color,thresold):
    imageCopy = image.copy()
    graImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        
    imageWidth = CannyImage.shape[1]
    imageHeight = CannyImage.shape[0]
    theta = np.deg2rad(np.arange(0.0, 180.0))
    
    #initializing the 2D accumulator
    # rows - 2* diagnonal length of the image
    # columns - 0 to 180
    accrHeight = int(np.around(np.sqrt(imageWidth*imageWidth + imageHeight*imageHeight),0))
    accr = np.zeros((2*accrHeight,len(theta)))
    rhoValues = np.linspace(-accrHeight,accrHeight,2*accrHeight)
    
    for i in range(0,imageHeight):
        for j in range(0,imageWidth):
            if CannyImage[i,j] > 10 :
                for k in range(0,len(theta)):
                    rho = int(np.around(i*(np.cos(theta[k])) + j*(np.sin(theta[k])),0)) 
                    accr[rho+accrHeight,k] += 1
          
    linesList = []
    #thersolding the number of votes
    for i in range(0,accr.shape[0]):
        for j in range(0,accr.shape[1]):
            if accr[i,j] > thresold:
                linesList.append([rhoValues[i],theta[j]])
                
    # plotting the blue lines on the given image using rho, theta values
    if color == "blue":
        for i in range(0,len(linesList)):
            temp = linesList[i]
            theta1 = temp[1]
            r = temp[0]
            if theta1 > 2.0:
                a = np.cos(theta1)
                b = np.sin(theta1)
                x1 = b*r
                y1 = a*r
                x2 = int(x1 + 1000*(a))
                y2 = int(y1 + 1000*(-b))
                x3 = int(x1 - 1000*(a))
                y3 = int(y1 - 1000*(-b))

                cv2.line(imageCopy,(x2,y2),(x3,y3),(255,0,0),1)
        
    # plotting the red lines on the given image using rho, theta values
    if color == "red":
        for i in range(0,len(linesList)):
            temp = linesList[i]
            theta1 = temp[1]
            r = temp[0]
            if theta1 < 2.0:
                a = np.cos(theta1)
                b = np.sin(theta1)
                x1 = b*r
                y1 = a*r
                x2 = int(x1 + 1000*(a))
                y2 = int(y1 + 1000*(-b))
                x3 = int(x1 - 1000*(a))
                y3 = int(y1 - 1000*(-b))
                cv2.line(imageCopy,(x2,y2),(x3,y3),(0,0,255),1)
        
    return imageCopy

# Sobel edge dectetion algorithm
def gradient_mag_cal(sobel_x,sobel_y,image):
    sobel_copy_x = sobel_x.copy()
    sobel_copy_y = sobel_y.copy()
    
    sobel_mag_image = np.zeros(image.shape)
    
    sobel_x_height = sobel_x.shape[0]
    sobel_y_width = sobel_y.shape[1]
    
    
    for i in range(sobel_x_height):
        for j in range(sobel_y_width):
            sobel_mag_image[i,j] = sobel_copy_x[i,j]*sobel_copy_x[i,j] + sobel_copy_y[i,j]*sobel_copy_y[i,j]
            sobel_mag_image[i,j] = np.sqrt(sobel_mag_image[i,j])
            
    return sobel_mag_image


# edge deection algorithm
def edgeDetectionAlgo(kernel_matrix_x,kernel_matrix_y,image):
    
    EdgeDetectedImage_x = conv(image,kernel_matrix_x)
    EdgeDetectedImage_y = conv(image,kernel_matrix_y)
    circleDetectedImage = cv2.Canny(image,10,250)
    circleDetectImage = gradient_mag_cal(EdgeDetectedImage_x,EdgeDetectedImage_y,image)
    circleDetectImage = np.abs(circleDetectImage) / np.max(np.abs(circleDetectImage))
    circleDetectImage = np.dot(255,circleDetectImage)
    circleDetectImage = circleDetectImage.astype('uint8')
    
    return circleDetectedImage


#hough circle implementation
def houghCircleImple(CannyImage,image,radius):
    image1 = image.copy()
    imageWidth  = CannyImage.shape[1]
    imageHeight = CannyImage.shape[0]
    theta = np.deg2rad(np.arange(0.0, 180.0))
    list1 = []
    list2 = []
    
    # intialzing the accumulator value
    accumulator = np.zeros((2*imageWidth,2*imageWidth))
    for i in range(0,imageHeight):
        for j in range(0,imageWidth):
            if CannyImage[i,j] > 10:
                for k in range(0,len(theta)):
                    a = int(np.around(i - radius*np.cos(theta[k]),0)) 
                    b = int(np.around(j - radius*np.sin(theta[k]),0)) 
                    list1.append(a)
                    list2.append(b)
                    accumulator[a+imageWidth,b+imageWidth] += 1
    circle = []
    # thersolding the voting of the accumulator
    for i in range(0,accumulator.shape[0]):
        for j in range(0,accumulator.shape[1]):
            if accumulator[i,j] > 96:
                circle.append([i-imageWidth,j-imageWidth])
    # plotting the circles on the given image
    for center in circle:
        temp = center
        center_x = temp[0]
        center_y = temp[1]
        cv2.circle(image1, (center_y, center_x), int(radius), (0, 255, 0), 4)
    return image1

# line detection algorithm
def lineDetectionAlgo(image,kernel):

    sobel_vertical = conv(image,kernel)
    sobel_vertical = np.abs(sobel_vertical) / np.max(np.abs(sobel_vertical))
    sobel_vertical = np.dot(255,sobel_vertical)
    sobel_vertical = sobel_vertical.astype('uint8')
    
    return sobel_vertical
    


# In[37]:


def main():
    image = cv2.imread('/Users/vidyach/Desktop/cvip/project3/original_imgs/hough.jpg',0)
    colorImage = cv2.imread('/Users/vidyach/Desktop/cvip/project3/original_imgs/hough.jpg')
     
    kernel_matrix_vertical = np.array([[-1,2,-1],[-1,2,-1],[-1,2,-1]], np.int32)
    kernel_matrix_inclined = np.array([[2,-1,-1],[-1,2,-1],[-1,-1,2]], np.int32)
    
    
    #callin the line dectection algorithm
    EdgeDetectedImageVertical = lineDetectionAlgo(image,kernel_matrix_vertical)
    EdgeDetectedImageInclined = lineDetectionAlgo(image,kernel_matrix_inclined)
    
    
   # calling the hough Tranform implementation function
    RedImage = HoughTImple(EdgeDetectedImageVertical,colorImage,"red",360)
    cv2.imwrite("/Users/vidyach/Desktop/cvip/project3/output/task3/red_line.jpg",RedImage)
    blueImage = HoughTImple(EdgeDetectedImageInclined,colorImage,"blue",220)
    cv2.imwrite("/Users/vidyach/Desktop/cvip/project3/output/task3/blue_line.jpg",blueImage)
main()


# # hough circle transform - detecting coins in images

# In[38]:


def main(): 
    image = cv2.imread('/Users/vidyach/Desktop/cvip/project3/original_imgs/hough.jpg',0)
    colorImage = cv2.imread('/Users/vidyach/Desktop/cvip/project3/original_imgs/hough.jpg')
    
    
    # applying edge detection algorithm on the input imaga
    kernel_matrix_x = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], np.int32)
    kernel_matrix_y = np.matrix([[1,2,1],[0,0,0],[-1,-2,-1]],np.int32)
    circleDetectedImage = edgeDetectionAlgo(kernel_matrix_x,kernel_matrix_y,image)
    
    # calling the hough circle implementation fucntion
    coins = houghCircleImple(circleDetectedImage,colorImage,21)
    
    cv2.imwrite("/Users/vidyach/Desktop/cvip/project3/output/task3/coin.jpg",coins)
main()

