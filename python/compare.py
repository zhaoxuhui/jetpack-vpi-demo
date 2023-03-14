import vpi
import cv2
import time

if __name__ == '__main__':
    img_path = "../imgs/genshin.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    input = vpi.asimage(img)

    t1 = time.time()
    with vpi.Backend.CUDA:
        output = input.box_filter(5, border=vpi.Border.ZERO)
    t2 = time.time()
    
    dt_vpi_cuda = 1000 * (t2 - t1)
    print("dt_vpi_cuda:", dt_vpi_cuda, "ms")

    t3 = time.time()
    output_blur = cv2.blur(img, (5, 5))
    t4 = time.time()
    
    dt_opencv = 1000 * (t4 - t3)
    print("dt_opencv:", dt_opencv, "ms")