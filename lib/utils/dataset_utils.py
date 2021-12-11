import dill
import PIL

def restore(data):
    """
    In case we dilled some structures to share between multiple process this function will restore them.
    If the data input are not bytes we assume it was not dilled in the first place

    :param data: Possibly dilled data structure
    :return: Un-dilled data structure
    """
    if type(data) is bytes:
        return dill.loads(data)
    return data
    
def squarify(bbox, squarify_ratio, img_width):
    width = abs(bbox[0] - bbox[2])
    height = abs(bbox[1] - bbox[3])
    width_change = height * squarify_ratio - width
    bbox[0] = bbox[0] - width_change/2
    bbox[2] = bbox[2] + width_change/2
    # Squarify is applied to bounding boxes in Matlab coordinate starting from 1
    if bbox[0] < 0:
        bbox[0] = 0
    
    # check whether the new bounding box goes beyond image boarders
    # If this is the case, the bounding box is shifted back
    if bbox[2] > img_width:
        bbox[0] = bbox[0]-bbox[2] + img_width
        bbox[2] = img_width
    return bbox

def img_pad(img, mode = 'warp', size = 224):
    '''
    Pads a given image.
    Crops and/or pads a image given the boundries of the box needed
    img: the image to be coropped and/or padded
    bbox: the bounding box dimensions for cropping
    size: the desired size of output
    mode: the type of padding or resizing. The modes are,
        warp: crops the bounding box and resize to the output size
        same: only crops the image
        pad_same: maintains the original size of the cropped box  and pads with zeros
        pad_resize: crops the image and resize the cropped box in a way that the longer edge is equal to
        the desired output size in that direction while maintaining the aspect ratio. The rest of the image is
        padded with zeros
        pad_fit: maintains the original size of the cropped box unless the image is biger than the size in which case
        it scales the image down, and then pads it
    '''
    assert(mode in ['same', 'warp', 'pad_same', 'pad_resize', 'pad_fit']), 'Pad mode %s is invalid' % mode
    image = img.copy()
    if mode == 'warp':
        warped_image = image.resize((size,size),PIL.Image.NEAREST)
        return warped_image
    elif mode == 'same':
        return image
    elif mode in ['pad_same','pad_resize','pad_fit']:
        img_size = image.size  # size is in (width, height)
        ratio = float(size)/max(img_size)
        if mode == 'pad_resize' or  \
            (mode == 'pad_fit' and (img_size[0] > size or img_size[1] > size)):
            img_size = tuple([int(img_size[0]*ratio),int(img_size[1]*ratio)])
            image = image.resize(img_size, PIL.Image.NEAREST)
        padded_image = PIL.Image.new("RGB", (size, size))
        padded_image.paste(image, ((size-img_size [0])//2,
                    (size-img_size [1])//2))
        return padded_image
