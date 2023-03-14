import vpi
import cv2

if __name__ == '__main__':
    img_path = "../imgs/genshin.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # step1 将读取的Numpy Array对象转换成VPI的影像格式
    input = vpi.asimage(img)

    # step2 开始执行模糊卷积
    # 对于任何一个VPI函数都需要显式指定backend(以函数参数方式指定或者以with方式指定)
    # box_filter是VPI的函数
    with vpi.Backend.CUDA:
        output = input.box_filter(5, border=vpi.Border.ZERO)

    # step3 结果输出
    with output.rlock_cpu() as outData:
        cv2.imwrite("../imgs/blurred_with_python.jpg", outData)