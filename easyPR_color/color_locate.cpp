#include<opencv2\opencv.hpp>
#include<iostream>

using namespace cv;
using namespace std;

enum color{ BLUE, YELLOW};
Mat colorMatch(const Mat &src,const color r,
	const bool adaptive_minsv) {

	// if use adaptive_minsv
	// min value of s and v is adaptive to h
	const float max_sv = 255;  //S<V 通道的阈值
	const float minref_sv = 64;
	const float minabs_sv = 95; //95;

     // H range of blue 
	const int min_blue = 100;  //200-280
	const int max_blue = 140;  

    // H range of yellow
	const int min_yellow = 15;  // 30-80
	const int max_yellow = 40;  


	Mat src_hsv;
	std::vector<Mat> hsvSplit;
	cvtColor(src, src_hsv, CV_BGR2HSV);
	split(src_hsv, hsvSplit);
	equalizeHist(hsvSplit[2], hsvSplit[2]);//对亮度进行直方图均衡
	merge(hsvSplit, src_hsv);
	imshow("src_hsv", src_hsv);
	// match to find the color

	int min_h = 0;
	int max_h = 0;
	switch (r) {
	case BLUE:
		min_h = min_blue;
		max_h = max_blue;
		break;
	case YELLOW:
		min_h = min_yellow;
		max_h = max_yellow;
		break;
	default:// Color::UNKNOWN
		break;
	}

	float diff_h = float((max_h - min_h) / 2);
	float avg_h = min_h + diff_h;

	int channels = src_hsv.channels();
	int nRows = src_hsv.rows;

	// consider multi channel image
	int nCols = src_hsv.cols * channels;
	if (src_hsv.isContinuous()) {
		nCols *= nRows;
		nRows = 1;
	}

	int i, j;
	uchar* p;
	float s_all = 0;
	float v_all = 0;
	float count = 0;
	for (i = 0; i < nRows; ++i) {
		p = src_hsv.ptr<uchar>(i);
		for (j = 0; j < nCols; j += 3) {
			int H = int(p[j]);      // 0-180
			int S = int(p[j + 1]);  // 0-255
			int V = int(p[j + 2]);  // 0-255

			s_all += S;
			v_all += V;
			count++;
			bool colorMatched = false;
			if (H > min_h && H < max_h) {
				float Hdiff = 0;
				if (H > avg_h)
					Hdiff = H - avg_h;
				else
					Hdiff = avg_h - H;

				float Hdiff_p = float(Hdiff) / diff_h; //diff_h = (max_h - min_h)/2=20

				float min_sv = 0;
				if (true == adaptive_minsv)
					min_sv =minref_sv -minref_sv / 2 *(1- Hdiff_p);  // inref_sv - minref_sv / 2 * (1 - Hdiff_p)
				else
					min_sv = minabs_sv;  // add

				if ((S > min_sv && S < max_sv) && (V > min_sv && V < max_sv))
					colorMatched = true;
			}

			if (colorMatched == true) {
				p[j] = 0;
				p[j + 1] = 0;
				p[j + 2] = 255;
			}
			else {
				p[j] = 0;
				p[j + 1] = 0;
				p[j + 2] = 0;
			}
		}
	}

	// cout << "avg_s:" << s_all / count << endl;
	// cout << "avg_v:" << v_all / count << endl;

	// get the final binary

	Mat src_grey;
	std::vector<Mat> hsvSplit_done;
	split(src_hsv, hsvSplit_done);
	src_grey = hsvSplit_done[2];
	return src_grey;
}

int main() {
	Mat src = imread("blueplane.jpg");
	imshow("src", src);
	Mat mask = colorMatch(src, BLUE, true);
	imshow("mask", mask);
	waitKey(0);

}