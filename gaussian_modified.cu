/* Compile and run: nvcc -arch=sm_35 -I/usr/local/include/opencv2/ `pkg-config --cflags opencv` -o gaussian_modified gaussian_modified.cu -I/usr/local/include/opencv2/ `pkg-config opencv --cflags --libs` -lopencv_gpu -lopencv_core -lopencv_highgui
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

	char s_append[] = "gaussian_modified";
	char input[100] = {0};
	char output_1[100] = {0};
	char output_2[100] = {0};

	int num_tests = 10;
	clock_t gpu_time;
	clock_t cpu_time;

	parse_input(argv, s_append, input, output_1, output_2);

	Mat input_cpu = imread(input, CV_LOAD_IMAGE_COLOR);
	Mat output_cpu;
	std::vector<Mat> bgr_planes;
	split(input_cpu, bgr_planes);

	for (int i = 0; i < num_tests; ++i) {
		cuda::GpuMat output_cuda;

		cuda::GpuMat input_cuda(bgr_planes[1]);
		clock_t tStart2 = clock();
		cv::Ptr<cv::cuda::Filter> filter = cv::cuda::createGaussianFilter(input_cuda.type(), output_cuda.type(), Size(3, 3), 0, 0);
		filter->apply(input_cuda, output_cuda);
		clock_t tEnd2 = clock();
		Mat output_cuda_cpu;
		output_cuda.download(output_cuda_cpu);

		clock_t tStart3 = clock();
		cv::GaussianBlur(bgr_planes[1], output_cpu, Size(3, 3), 0, 0);
		clock_t tEnd3 = clock();

		//imwrite(output_1, output_cuda_cpu);
		//imwrite(output_2, output_cpu);
		//imshow("Result", input_cpu);
		//cv::waitKey();
		//imshow("Result", output_cuda_cpu);
		//cv::waitKey();
		//imshow("Result", output_cpu);
		//cv::waitKey();

		input_cuda.release();
		output_cuda.release();
		output_cpu.release();

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
