#include <iostream>

#include <vpi/OpenCVInterop.hpp>
#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/algo/BoxFilter.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {
    // 第一阶段：初始化
    // step1 利用OpenCV读取影像
    string img_path = "../../imgs/genshin.jpg";
    Mat img = imread(img_path, IMREAD_GRAYSCALE);

    // step2 创建stream，第一个参数是backend，如果指定为0则表示可以在任何backend执行
    VPIStream stream;
    vpiStreamCreate(0, &stream);

    // step3 基于读取的影像构造VPIImage对象
    VPIImage image;
    vpiImageCreateWrapperOpenCVMat(img, 0, &image);

    // step4 新建VPIImage变量用于储存模糊结果
    VPIImage blurred;
    vpiImageCreate(img.cols, img.rows, VPI_IMAGE_FORMAT_U8, 0, &blurred);

    // 第二阶段：执行
    time_t t1 = clock();

    // step1 开始滤波
    vpiSubmitBoxFilter(stream, VPI_BACKEND_CUDA, image, blurred, 5, 5, VPI_BORDER_ZERO);

    // 等待所有操作执行完成
    vpiStreamSync(stream);
    time_t t2 = clock();
    double dt = 1000 * (double) (t2 - t1) / CLOCKS_PER_SEC;
    cout << "vpi cost time: " << dt << " ms" << endl;

    // step2 锁定blurred对象，取出数据
    VPIImageData outData;
    vpiImageLockData(blurred, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &outData);

    // step3 将取出的数据转换为OpenCV格式并保存
    Mat out_mat;
    vpiImageDataExportOpenCVMat(outData, &out_mat);
    imwrite("../../imgs/blurred_with_cpp.jpg", out_mat);

    // step4 解除对于blurred对象的锁定
    vpiImageUnlock(blurred);

    // 第三阶段：清理
    vpiStreamDestroy(stream);
    vpiImageDestroy(image);
    vpiImageDestroy(blurred);

    // 额外步骤，对比OpenCV的速度
    Mat out_img_opencv;
    time_t t3 = clock();
    blur(img, out_img_opencv, Size(5, 5));
    time_t t4 = clock();
    double dt2 = 1000 * (double) (t4 - t3) / CLOCKS_PER_SEC;
    cout << "opencv cost time: " << dt2 << " ms" << endl;

    return 0;
}
