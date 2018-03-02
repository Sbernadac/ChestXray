import numpy as np
import cv2
import time
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.models import Sequential, load_model
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout
from scipy.misc import imsave

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    print("1x.shape="+str(x.shape))
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    print("2x.shape="+str(x.shape))
    #if x.shape[2] != 3:
        #x = x.transpose((1, 2, 0))
    #print("3x.shape="+str(x.shape))
    x = np.clip(x, 0, 255).astype('uint8')
    print("4x.shape="+str(x.shape))
    return x


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


#Define regularizations:
def blur_regularization(img, grads, size = (3, 3)):
    return cv2.blur(img, size)

def decay_regularization(img, grads, decay = 0.8):
    return decay * img

def clip_weak_pixel_regularization(img, grads, percentile = 1):
    clipped = img
    threshold = np.percentile(np.abs(img), percentile)
    clipped[np.where(np.abs(img) < threshold)] = 0
    return clipped

if __name__ == "__main__":


    #Configuration:
    img_width, img_height = 244 , 244
    input_shape = (img_width, img_height,1)
    #cropping dimension
    crop_x,crop_y,crop_w,crop_h=(112,112,800,800)

    #filter_indexes = range(0, 10)
    filter_index = 0

    #input_placeholder = K.placeholder((1, 3, img_width, img_height))
    #first_layer = ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height))
    #first_layer.input = input_placeholder


    model = load_model('myAtelectasis.h5')
    input_img = model.input
    #layer = get_output_layer(model, 'conv2d_5')
    #layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_name = 'conv2d_2'


    img = cv2.imread('../../images/00026568_000.png', 0)
    img = img[crop_x:crop_x+crop_w,crop_y:crop_y+crop_h]
    img = cv2.resize(img, (img_width, img_height))
     #init_img = [np.transpose(img, (2, 0, 1))]
    img = img_to_array(img)


    kept_filters = []
    for filter_index in range(256):
        # we only scan through the first 200 filters,
        # but there are actually 512 of them
        print('Processing filter %d' % filter_index)
        start_time = time.time()

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = layer_dict[layer_name].output
        print("layer_output.shape : "+str(layer_output.shape))
        loss = K.mean(layer_output[:, :, :, filter_index])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.

        # we start from a gray image with some random noise
        input_img_data = np.random.random((1, img_width, img_height, 1))
        input_img_data = (input_img_data - 0.5) * 20 + 128

        # we run gradient ascent for 20 steps
        for i in range(200):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            print('Current loss value:', loss_value)
            if loss_value <= 0.:
                # some filters get stuck to 0, we can skip them
                break

        # decode the resulting input image
        if loss_value > 0:
            img = deprocess_image(input_img_data[0])
            print("loss_value img.shape="+str(img.shape))
            kept_filters.append((img, loss_value))
        end_time = time.time()
        print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

    # we will stich the best 64 filters on a 8 x 8 grid.
    n = 4

    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top 64 filters.
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

        # build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 1))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            #print("img.shape="+str(img.shape))
            cv2.imwrite("stitched"+str(i*n)+"_"+str(j)+".png",img)
            stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                         (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    # save the result to disk
    #imsave('stitched_filters_%dx%d.png' % (n, n), stitched_filters)
    
