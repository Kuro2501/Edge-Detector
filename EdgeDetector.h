#pragma once
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "Convolution.h"
#include <math.h>
#include <iostream>
using namespace cv;
using namespace std;
class EdgeDetector
{
public:
	/*
	Hàm phát hiện biên cạnh của ảnh xám theo phương pháp Sobel
	sourceImage: ảnh input
	destinationImage: ảnh output
	Hàm trả về
		0: nếu detect thành công
		1: nếu detect thất bại (không đọc được ảnh input,...)
	*/
	int detectBySobel(const Mat& sourceImage, Mat& destinationImage) {
		if (sourceImage.empty())
			return 1;

		Convolution convolution;
		vector<float> kernelX = { -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0 };
		vector<float> kernelY = { -1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0 };

		for (int i = 0; i < 9; i++) {
			kernelX[i] *= 1.0 / 4;
			kernelY[i] *= 1.0 / 4;
		}

		Mat dstImageX, dstImageY;

		int width = sourceImage.cols;
		int height = sourceImage.rows;
		int nChannels = sourceImage.channels();
		destinationImage.create(height, width, sourceImage.type());
		int widthStep = sourceImage.step[0];

		uchar* pSrc = (uchar*)sourceImage.data;
		uchar* pDst = (uchar*)destinationImage.data;
		
		uchar* pDstRow, * pDstXRow, * pDstYRow;
		
		convolution.SetKernel(kernelX, 3, 3);
		convolution.DoConvolution(sourceImage, dstImageX);
		convolution.SetKernel(kernelY, 3, 3);
		convolution.DoConvolution(sourceImage, dstImageY);

		uchar* pDstX = (uchar*)dstImageX.data;
		uchar* pDstY = (uchar*)dstImageY.data;
		for (int y = 0; y < height; y++, pSrc += widthStep, pDst += widthStep, pDstX += widthStep, pDstY += widthStep) {
			const uchar* pSrcRow = pSrc;
			pDstRow = pDst;
			pDstXRow = pDstX;
			pDstYRow = pDstY;
			for (int x = 0; x < width; x++, pSrcRow += nChannels, pDstRow += nChannels, pDstXRow += nChannels, pDstYRow += nChannels) {
				//pDstRow[0] = pSrcRow[0] + pDstXRow[0] + pDstYRow[0];
				pDstRow[0] = pDstXRow[0] + pDstYRow[0];
			}
		}
		imshow("Sobel X Image", dstImageX);
		imshow("Sobel Y Image", dstImageY);
	}

	/*
	Hàm phát hiện biên cạnh của ảnh xám theo phương pháp Prewitt
	sourceImage: ảnh input
	destinationImage: ảnh output
	Hàm trả về
		0: nếu detect thành công
		1: nếu detect thất bại (không đọc được ảnh input,...)
	*/
	int detectByPrewitt(const Mat& sourceImage, Mat& destinationImage) {
		if (sourceImage.empty())
			return 1;

		Convolution convolution;
		vector<float> kernelX = { -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0 };
		vector<float> kernelY = { -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0 };

		for (int i = 0; i < 9; i++) {
			kernelX[i] *= 1.0 / 4;
			kernelY[i] *= 1.0 / 4;
		}

		Mat dstImageX, dstImageY;

		int width = sourceImage.cols;
		int height = sourceImage.rows;
		int nChannels = sourceImage.channels();
		destinationImage.create(height, width, sourceImage.type());
		int widthStep = sourceImage.step[0];

		uchar* pSrc = (uchar*)sourceImage.data;
		uchar* pDst = (uchar*)destinationImage.data;

		uchar* pDstRow, * pDstXRow, * pDstYRow;

		convolution.SetKernel(kernelX, 3, 3);
		convolution.DoConvolution(sourceImage, dstImageX);
		convolution.SetKernel(kernelY, 3, 3);
		convolution.DoConvolution(sourceImage, dstImageY);

		uchar* pDstX = (uchar*)dstImageX.data;
		uchar* pDstY = (uchar*)dstImageY.data;
		for (int y = 0; y < height; y++, pSrc += widthStep, pDst += widthStep, pDstX += widthStep, pDstY += widthStep) {
			const uchar* pSrcRow = pSrc;
			pDstRow = pDst;
			pDstXRow = pDstX;
			pDstYRow = pDstY;
			for (int x = 0; x < width; x++, pSrcRow += nChannels, pDstRow += nChannels, pDstXRow += nChannels, pDstYRow += nChannels) {
				//pDstRow[0] = pSrcRow[0] + pDstXRow[0] + pDstYRow[0];
				pDstRow[0] = pDstXRow[0] + pDstYRow[0];
			}
		}
		imshow("Prewitt X Image", dstImageX);
		imshow("Prewitt Y Image", dstImageY);
	}

	/*
	Hàm phát hiện biên cạnh của ảnh xám theo phương pháp Laplace
	sourceImage: ảnh input
	destinationImage: ảnh output
	Hàm trả về
		0: nếu detect thành công
		1: nếu detect thất bại (không đọc được ảnh input,...)
	*/
	int detectByLaplace(const Mat& sourceImage, Mat& destinationImage) {
		if (sourceImage.empty())
			return 1;

		Convolution convolution;
		vector<float> kernelLap = { 0.0, 1.0, 0.0, 1.0, -4.0, 1.0, 0.0, 1.0, 0.0 };

		Mat dstImageLap;

		int width = sourceImage.cols;
		int height = sourceImage.rows;
		int nChannels = sourceImage.channels();
		destinationImage.create(height, width, sourceImage.type());
		int widthStep = sourceImage.step[0];

		uchar* pSrc = (uchar*)sourceImage.data;
		uchar* pDst = (uchar*)destinationImage.data;

		uchar* pDstRow, * pDstLapRow;

		convolution.SetKernel(kernelLap, 3, 3);
		convolution.DoConvolution(sourceImage, dstImageLap);

		uchar* pDstLap = (uchar*)dstImageLap.data;
		for (int y = 0; y < height; y++, pSrc += widthStep, pDst += widthStep, pDstLap += widthStep) {
			const uchar* pSrcRow = pSrc;
			pDstRow = pDst;
			pDstLapRow = pDstLap;
			for (int x = 0; x < width; x++, pSrcRow += nChannels, pDstRow += nChannels, pDstLapRow += nChannels) {
				//pDstRow[0] = pSrcRow[0] + pDstXRow[0] + pDstYRow[0];
				pDstRow[0] = pDstLapRow[0];
			}
		}
	}

	/*
	Hàm phát hiện biên cạnh của ảnh xám theo phương pháp Canny
	sourceImage: ảnh input
	destinationImage: ảnh output
	Hàm trả về
		0: nếu detect thành công
		1: nếu detect thất bại (không đọc được ảnh input,...)
	*/
	int detectByCany(const Mat& sourceImage, Mat& destinationImage, int minVal, int maxVal) {
		if (sourceImage.empty())
			return 1;
		Convolution convolution;
		Mat srcBlur;

		// Giảm nhiễu bằng bộ lọc Gaussian 5x5
		vector<float> kernelGauss = { 2.0, 4.0, 5.0, 4.0, 2.0, 
									  4.0, 9.0, 12.0, 9.0, 4.0,
									  5.0, 12.0, 15.0, 12.0, 5.0,
									  4.0, 9.0, 12.0, 9.0, 4.0,
									  2.0, 4.0, 5.0, 4.0, 2.0 };

		for (int i = 0; i < 25; i++) {
			kernelGauss[i] *= 1.0 / 159;
		}
		convolution.SetKernel(kernelGauss, 5, 5);
		convolution.DoConvolution(sourceImage, srcBlur);
		//imshow("Blur Image", srcBlur);
		
		// Tìm Intensity Gradient và hướng Gradient
		vector<float> kernelX = { -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0 };
		vector<float> kernelY = { -1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0 };

		for (int i = 0; i < 9; i++) {
			kernelX[i] *= 1.0 / 4;
			kernelY[i] *= 1.0 / 4;
		}

		Mat dstImageX, dstImageY;
		Mat angle;

		int width = sourceImage.cols;
		int height = sourceImage.rows;
		int nChannels = sourceImage.channels();
		destinationImage.create(height, width, sourceImage.type());
		angle.create(height, width, sourceImage.type());
		int widthStep = sourceImage.step[0];

		uchar* pSrc = (uchar*)srcBlur.data;
		uchar* pDst = (uchar*)destinationImage.data;
		uchar* pAngle = (uchar*)angle.data;

		uchar* pDstRow, * pDstXRow, * pDstYRow, * pAngleRow;

		convolution.SetKernel(kernelX, 3, 3);
		convolution.DoConvolution(srcBlur, dstImageX);
		convolution.SetKernel(kernelY, 3, 3);
		convolution.DoConvolution(srcBlur, dstImageY);

		uchar* pDstX = (uchar*)dstImageX.data;
		uchar* pDstY = (uchar*)dstImageY.data;
		for (int y = 0; y < height; y++, pSrc += widthStep, pDst += widthStep, pDstX += widthStep, pDstY += widthStep, pAngle += widthStep) {
			const uchar* pSrcRow = pSrc;
			pDstRow = pDst;
			pDstXRow = pDstX;
			pDstYRow = pDstY;
			pAngleRow = pAngle;
			for (int x = 0; x < width; x++, pSrcRow += nChannels, pDstRow += nChannels, pDstXRow += nChannels, pDstYRow += nChannels, pAngleRow += nChannels) {
				pDstRow[0] = sqrt(pDstXRow[0] * pDstXRow[0] + pDstYRow[0] * pDstYRow[0]);
				pAngleRow[0] = atan2(pDstYRow[0], pDstXRow[0]) * 180 / 3.141592654;

				if ((float)pAngleRow[0] >= 0 && (float)pAngleRow[0] <= 22) pAngleRow[0] = 0;
				if ((float)pAngleRow[0] >= 23 && (float)pAngleRow[0] <= 67) pAngleRow[0] = 45;
				if ((float)pAngleRow[0] >= 68 && (float)pAngleRow[0] <= 112) pAngleRow[0] = 90;
				if ((float)pAngleRow[0] >= 113 && (float)pAngleRow[0] <= 157) pAngleRow[0] = 135;		
				//cout << (float)pAngleRow[0] << endl;
			}
		}

		// Non-maximum Suppression
		pAngle = (uchar*)angle.data;
		for (int y = 0; y < height; y++, pAngle += widthStep) {
			pAngleRow = pAngle;
			for (int x = 0; x < width; x++, pAngleRow += nChannels) {
				if ((float)pAngleRow[0] == 0) {
					if ((float)pAngleRow[0] < (float)pAngleRow[-1] || (float)pAngleRow[0] < (float)pAngleRow[1]) pAngleRow[0] = 0;
				}
				if ((float)pAngleRow[0] == 45) {
					if ((float)pAngleRow[0] < (float)pAngleRow[-widthStep + 1] || (float)pAngleRow[0] < (float)pAngleRow[widthStep - 1]) pAngleRow[0] = 0;
				}
				if ((float)pAngleRow[0] == 90) {
					if ((float)pAngleRow[0] < (float)pAngleRow[-widthStep] || (float)pAngleRow[0] < (float)pAngleRow[widthStep]) pAngleRow[0] = 0;
				}
				if ((float)pAngleRow[0] == 135) {
					if ((float)pAngleRow[0] < (float)pAngleRow[-widthStep - 1] || (float)pAngleRow[0] < (float)pAngleRow[widthStep + 1]) pAngleRow[0] = 0;
				}
			}
		}

		// Lọc ngưỡng Threshold
		pAngle = (uchar*)angle.data;
		pDst = (uchar*)destinationImage.data;
		for (int y = 0; y < height; y++, pAngle += widthStep, pDst += widthStep) {
			pAngleRow = pAngle;
			pDstRow = pDst;
			for (int x = 0; x < width; x++, pAngleRow += nChannels, pDstRow += nChannels) {
				if ((float)pAngleRow[0] < minVal) pDstRow[0] = 0;
			}
		}

		vector<int> offsets;
		for (int y = -1; y <= 1; y++)
			for (int x = -1; x <= 1; x++)
				offsets.push_back(y * widthStep + x);

		pAngle = (uchar*)angle.data;
		pDst = (uchar*)destinationImage.data;
		for (int y = 0; y < height; y++, pAngle += widthStep, pDst += widthStep) {
			pAngleRow = pAngle;
			pDstRow = pDst;
			for (int x = 0; x < width; x++, pAngleRow += nChannels, pDstRow += nChannels) {
				if ((float)pAngleRow[0] >= minVal && (float)pAngleRow[0] < maxVal) {
					bool isAdjacent = false;
					for (int i = 0; i < 9; i++) {
						if ((float)pDstRow[offsets[i]] != 0) isAdjacent = true;
					}
					if (isAdjacent == false) pDstRow[0] = 0;
				}
			}
		}
	}

	EdgeDetector() {};
	~EdgeDetector() {};
};

