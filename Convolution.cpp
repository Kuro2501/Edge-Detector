#include "Convolution.h"

vector<float> Convolution::GetKernel() {
	return this->_kernel;
}

void Convolution::SetKernel(vector<float> kernel, int kWidth, int kHeight) {
	this->_kernel = kernel;
	this->_kernelWidth = kWidth;
	this->_kernelHeight = kHeight;
}

int Convolution::DoConvolution(const Mat& sourceImage, Mat& destinationImage) {
	if (sourceImage.empty())
		return 1;
	int width = sourceImage.cols;
	int height = sourceImage.rows;
	int nChannels = sourceImage.channels();
	destinationImage.create(height, width, CV_8UC1);

	if (nChannels == 3)
		return 1;

	int xStart = _kernelWidth / 2, yStart = _kernelHeight / 2;
	int xEnd = width - (_kernelWidth - 1), yEnd = height - (_kernelHeight - 1);
	int widthStep = sourceImage.step[0];

	uchar* pDst = (uchar*)destinationImage.data + yStart * widthStep + xStart;
	uchar* pSrc = (uchar*)sourceImage.data + yStart * widthStep + xStart;

	// Tạo bảng offsets
	int widthLimit = _kernelWidth / 2;
	int heightLimit = _kernelHeight / 2;
	vector<int> offsets;
	for (int y = -heightLimit; y <= heightLimit; y++)
		for (int x = -widthLimit; x <= widthLimit; x++)
			offsets.push_back(y * widthStep + x);


	// Tính tích chập
	int size = _kernelWidth * _kernelHeight;
	for (int y = yStart; y < yEnd; y++, pSrc += widthStep, pDst += widthStep) {
		const uchar* pSrcRow = pSrc;
		uchar* pDstRow = pDst;
		for (int x = xStart; x < xEnd; x++, pSrcRow++, pDstRow++) {
			float sum = 0;
			for (int i = 0; i < size; i++) 
				sum += pSrcRow[offsets[i]] * _kernel[i];
			if (sum < 0) sum = -sum;
			*pDstRow = (uchar)sum;
		}
	}
	return 0;
}

Convolution::Convolution() {}
Convolution::~Convolution() {}
