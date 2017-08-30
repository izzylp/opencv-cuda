/* Compile and run: nvcc -arch=sm_35 -I/usr/local/include/opencv2/ `pkg-config --cflags opencv` -O3 -o histogram histogram.cu -I/usr/local/include/opencv2/ `pkg-config opencv --cflags --libs` -lopencv_gpu -lopencv_core -lopencv_highgui
 */

#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/cudaarithm.hpp"
#include "parse_input.h"

using namespace cv;

int main(int argc, const char* argv[])
{
	if (argc != 2) {
		printf("Error\n");
		return 0;
	}

	char s_append[] = "histogram";
	char input[100] = {0};
	char output_1[100] = {0};
	char output_2[100] = {0};

	int num_tests = 10;
	clock_t gpu_time;
	clock_t cpu_time;

	parse_input(argv, s_append, input, output_1, output_2);
	Mat input_cpu = imread(input, CV_LOAD_IMAGE_COLOR);

	for (int i = 0; i < num_tests; ++i) {
		std::vector<Mat> bgr_planes;
		split(input_cpu, bgr_planes);

		Mat hist_r_cuda, hist_g_cuda, hist_b_cuda, histImg_cuda;

		// create colors channels
		cuda::GpuMat colorGpu_b(bgr_planes[0]);
		cuda::GpuMat colorGpu_g(bgr_planes[1]);
		cuda::GpuMat colorGpu_r(bgr_planes[2]);
		cuda::GpuMat histGpu_r;
		cuda::GpuMat histGpu_g;
		cuda::GpuMat histGpu_b;

		clock_t tStart2 = clock();
		cv::cuda::calcHist(colorGpu_r, histGpu_r);
		cv::cuda::calcHist(colorGpu_g, histGpu_g);
		cv::cuda::calcHist(colorGpu_b, histGpu_b);
		clock_t tEnd2 = clock();

		histGpu_r.download(hist_r_cuda);
		histGpu_g.download(hist_g_cuda);
		histGpu_b.download(hist_b_cuda);

		/* Establish the number of bins */
		int histSize = 256;

		/* Set the ranges (for B,G,R) */
		float range[] = {0, 256};
		const float* histRange = {range};

		bool uniform = true;
		bool accumulate = false;

		Mat b_hist, g_hist, r_hist;

		/* Compute the histograms */
		clock_t tStart3 = clock();
		calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
		calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
		calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
		clock_t tEnd3 = clock();

		/* Draw the histograms for B, G and R */
		//int hist_w = 512; 
		//int hist_h = 400;
		//int bin_w = cvRound((double) hist_w / histSize);

		//Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));
		//Mat histImage_cuda(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));

		/* Normalize the result to [0, histImage.rows] */
		//normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		//normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
		//normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

		//normalize(hist_b_cuda, hist_b_cuda, 0, histImage_cuda.rows, NORM_MINMAX, -1, Mat());
		//normalize(hist_g_cuda, hist_g_cuda, 0, histImage_cuda.rows, NORM_MINMAX, -1, Mat());
		//normalize(hist_r_cuda, hist_r_cuda, 0, histImage_cuda.rows, NORM_MINMAX, -1, Mat());

		/* Draw for each channel */
		//for (int i = 1; i < histSize; ++i) {
		//	line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
		//			Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
		//			Scalar(255, 0, 0), 2, 8, 0);
		//	line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
		//			Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
		//			Scalar(0, 255, 0), 2, 8, 0);
		//	line(histImage, Point(bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i - 1))),
		//			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
		//			Scalar(0, 0, 255), 2, 8, 0);

		//	line(histImage_cuda, Point(bin_w * (i - 1), hist_h - cvRound(hist_b_cuda.at<int>(i - 1))),
		//			Point(bin_w * (i), hist_h - cvRound(hist_b_cuda.at<int>(i))),
		//			Scalar(255, 0, 0), 2, 8, 0);
		//	line(histImage_cuda, Point(bin_w * (i - 1), hist_h - cvRound(hist_g_cuda.at<int>(i - 1))),
		//			Point(bin_w * (i), hist_h - cvRound(hist_g_cuda.at<int>(i))),
		//			Scalar(0, 255, 0), 2, 8, 0);
		//	line(histImage_cuda, Point(bin_w*(i-1), hist_h - cvRound(hist_r_cuda.at<int>(i - 1))),
		//			Point(bin_w*(i), hist_h - cvRound(hist_r_cuda.at<int>(i))),
		//			Scalar(0, 0, 255), 2, 8, 0);
		//}

		/* Display */
		//imwrite(output_1, histImage_cuda);
		//imwrite(output_2, histImage);

		colorGpu_r.release();
		colorGpu_g.release();
		colorGpu_b.release();
		histGpu_r.release();
		histGpu_g.release();
		histGpu_b.release();

		if (i == 0) {
			gpu_time = tEnd2 - tStart2;
			cpu_time = tEnd3 - tStart3;
		} else {
			gpu_time += tEnd2 - tStart2;
			cpu_time += tEnd3 - tStart3;
		}
	}

	input_cpu.release();

	printf("Time taken GPU: %.5f s\n", (double) gpu_time / CLOCKS_PER_SEC / num_tests);
	printf("Time taken CPU: %.5f s\n", (double) cpu_time / CLOCKS_PER_SEC / num_tests);
}
