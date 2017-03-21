#include <iostream>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

// function declarations
void renderHistogram(std::string caption, std::vector<double> hist);
std::vector<double> getHistogramData(cv::Mat image);

// function implementations
void renderHistogram(std::string caption, std::vector<double> hist) {
	// set histogram drawing area height
	int canvasHeight = 130;

	cv::Mat canvas = cv::Mat(canvasHeight, 256, CV_8UC3, 0.0);

	std::vector<double> normalisedHist(256, 0.0);

	// obtain max value for normalization later
	double max = *std::max_element(std::begin(hist), std::end(hist));

	// normalize the histogram (0, 1)
	for (int i = 0; i < 256; i++) {
		normalisedHist[i] = hist[i] / max;
		cv::line(canvas, cv::Point(i, 0), cv::Point(i, canvasHeight * normalisedHist[i]), cv::Scalar(255, 255, 255));
	}

	// need to be flipped since the histogram
	// drawing has to be from bottom to top 
	cv::flip(canvas, canvas, 0);

	cv::imshow(caption, canvas);
}

std::vector<double> getHistogramData(cv::Mat image) {
	std::vector<double> histData(256, 0.0);

	// count frequency of each gray levels
	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			double grayLevel = image.at<uchar>(row, col);
			histData[grayLevel] += 1;
		}
	}

	return histData;
}

cv::Mat getEqualizedImage(cv::Mat image) {
	std::vector<double> histData;
	cv::Mat result = image.clone();

	// initialize probabilities and cdf
	std::vector<double> pmf(256, 0.0);
	std::vector<double> cdf(256, 0.0);

	// getting gray level frequencies from original image
	histData = getHistogramData(image);
	
	// calculate probabilities of all gray levels
	for (int i = 0; i < 256; i++) {
		pmf[i] = histData[i] / (image.rows*image.cols);
	}

	// calculate cumulative distribution function
	cdf[0] = pmf[0];
	for (int i = 1; i < 256; i++) {
		cdf[i] = cdf[i - 1] + pmf[i];
	}

	// mapping into new grayscale intensity
	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			int intensity = image.at<uchar>(row, col);
			result.at<uchar>(row, col) = std::floor(255 * cdf[intensity]);
		}
	}
	
	return result;
}

void main() {
	cv::Mat image = cv::imread("d:/opencv/Lenna.png", CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat equalized = getEqualizedImage(image);

	cv::imshow("Original", image);
	cv::imshow("Equalized", equalized);

	renderHistogram("Hist. Original", getHistogramData(image));
	renderHistogram("Hist. Equalized", getHistogramData(equalized));

	cv::imwrite("d:/opencv/histeqOri.jpg", image);
	cv::imwrite("d:/opencv/histeqEq.jpg", equalized);

	cv::waitKey();
}