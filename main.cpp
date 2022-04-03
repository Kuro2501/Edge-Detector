#include "EdgeDetector.h"

int main() {
	Mat srcImage = imread("/Users/ThaiBinh/Downloads/Picture/test/4.png", IMREAD_GRAYSCALE);
	Mat dstImage;
	EdgeDetector edgeDetector;
	//edgeDetector.detectBySobel(srcImage, dstImage);
	//edgeDetector.detectByPrewitt(srcImage, dstImage);
	//edgeDetector.detectByLaplace(srcImage, dstImage);
	//edgeDetector.detectByCany(srcImage, dstImage, 40, 135);
	Canny(srcImage, dstImage, 40, 135);
	imshow("Source Image", srcImage);
	imshow("Destination Image", dstImage);

	waitKey(0);
}