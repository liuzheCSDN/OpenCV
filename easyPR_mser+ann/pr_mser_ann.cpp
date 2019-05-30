
// Mser车牌目标检测
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include<opencv2\ml\ml.hpp>
using namespace cv;
using namespace std;


// ！获取垂直和水平方向直方图
Mat ProjectedHistogram(Mat img, int t)
{
	int sz = (t) ? img.rows : img.cols;
	Mat mhist = Mat::zeros(1, sz, CV_32F);

	for (int j = 0; j<sz; j++) {
		Mat data = (t) ? img.row(j) : img.col(j);
		mhist.at<float>(j) = countNonZero(data);	//统计这一行或一列中，非零元素的个数，并保存到mhist中
	}
	//Normalize histogram
	double min, max;
	minMaxLoc(mhist, &min, &max);
	if (max>0)
		mhist.convertTo(mhist, -1, 1.0f / max, 0);//用mhist直方图中的最大值，归一化直方图

	return mhist;
}
//! 获得车牌的特征数
Mat getTheFeatures(Mat in)
{
	const int VERTICAL = 0;
	const int HORIZONTAL = 1;
	//Histogram features
	Mat vhist = ProjectedHistogram(in, VERTICAL);
	Mat hhist = ProjectedHistogram(in, HORIZONTAL);
	//Last 10 is the number of moments components
	int numCols = vhist.cols + hhist.cols;
	Mat out = Mat::zeros(1, numCols, CV_32F);
	//Asign values to feature,样本特征为水平、垂直直方图
	int j = 0;
	for (int i = 0; i<vhist.cols; i++)
	{
		out.at<float>(j) = vhist.at<float>(i);
		j++;
	}
	for (int i = 0; i<hhist.cols; i++)
	{
		out.at<float>(j) = hhist.at<float>(i);
		j++;
	}
	return out;
}


// ! EasyPR的getFeatures回调函数！本函数是获取垂直和水平的直方图图值
void getHistogramFeatures(const Mat& image, Mat& features)
{
	features = getTheFeatures(image);
}






std::vector<cv::Rect> mserGetPlate(cv::Mat srcImage)
{
	// HSV空间转换
	cv::Mat gray, gray_neg;
	// 灰度转换 
	cv::cvtColor(srcImage, gray, CV_BGR2GRAY);
	imshow("gray", gray);
	// 取反值灰度
	gray_neg = 255 - gray;
	std::vector<vector<Point> > regContours;
	std::vector<vector<Point> > charContours;//点集

	// 创建MSER对象
	int imageArea = gray.rows * gray.cols;
	int delta = 1;//const int delta = CParams::instance()->getParam2i();;
	const int minArea = 30;
	double maxAreaRatio = 0.001;

	cv::Ptr<cv::MSER> mesr1 = cv::MSER::create(delta, minArea, int(maxAreaRatio * imageArea), 0.15, 10);
	cv::Ptr<cv::MSER> mesr2 = cv::MSER::create(delta, minArea, 400, 0.1, 0.3);
	std::vector<cv::Rect> bboxes1;
	std::vector<cv::Rect> bboxes2;
	// MSER+ 检测
	mesr1->detectRegions(gray, regContours, bboxes1);
	// MSER-操作
	mesr2->detectRegions(gray_neg, charContours, bboxes2);

	cv::Mat mserMapMat = cv::Mat::zeros(srcImage.size(), CV_8UC1);
	cv::Mat mserNegMapMat = cv::Mat::zeros(srcImage.size(), CV_8UC1);

	for (int i = (int)regContours.size() - 1; i >= 0; i--)
	{
		// 根据检测区域点生成mser+结果
		const std::vector<cv::Point>& r = regContours[i];
		for (int j = 0; j < (int)r.size(); j++)
		{
			cv::Point pt = r[j];
			mserMapMat.at<unsigned char>(pt) = 255;
		}
	}
	//MSER- 检测
	for (int i = (int)charContours.size() - 1; i >= 0; i--)
	{
		// 根据检测区域点生成mser-结果
		const std::vector<cv::Point>& r = charContours[i];
		for (int j = 0; j < (int)r.size(); j++)
		{
			cv::Point pt = r[j];
			mserNegMapMat.at<unsigned char>(pt) = 255;
		}
	}
	imshow("mserMapMat", mserMapMat);
	//imshow("mserNegMapMat", mserNegMapMat);
	cv::Mat mserResMat;
	mserResMat = mserMapMat;
	mserResMat = mserMapMat & mserNegMapMat;	// mser+与mser-位与操作
	//imshow("mserResMat", mserResMat);
	
	
	// 寻找外部轮廓
	std::vector<std::vector<Point> > plate_contours;
	cv::findContours(mserMapMat, plate_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	// 候选车牌区域判断输出
	std::vector<cv::Rect> candidates;
	for (size_t i = 0; i != plate_contours.size(); ++i)
	{
		// 求解最小外界矩形
		cv::Rect rect = cv::boundingRect(plate_contours[i]);
		// 宽高比例
		double wh_ratio = rect.width / double(rect.height);
		if ( wh_ratio > 0.2 && wh_ratio < 0.9)
			candidates.push_back(rect);
	}
	return  candidates;
}
int main()
{
	cv::Mat srcImage =
		cv::imread("plate1.jpg");
	if (srcImage.empty())
		return-1;
	cv::imshow("src Image", srcImage);
	// 候选车牌区域检测
	std::vector<cv::Rect> candidates;
	candidates = mserGetPlate(srcImage);

	// 车牌区域显示
	for (int i = 0; i < candidates.size(); ++i)
	{
		string image = "rect" + cv::format("%.4d", i);
		namedWindow(image, 0);
		cv::imshow(image, srcImage(candidates[i]));
	}
	//ANN判别
	Ptr<ml::ANN_MLP> ann = cv::ml::ANN_MLP::load("ann.xml");// 120 40 65
	Mat feature,gray,feature2,dst;
	Mat out(1, 65, CV_32F);
	cvtColor(srcImage, gray, CV_BGR2GRAY);
	std::vector<cv::Rect> candidates2;
	for (int i = 0; i < candidates.size(); ++i)
	{
		Mat result = gray(candidates[i]);
		resize(result, dst, Size(10, 10));
		getHistogramFeatures(dst, feature);
		Mat dst2=dst.reshape(0, 1);//cn: 表示通道数, 如果设为0，则表示保持通道数不变，否则则变为设置的通道数。
                                   //rows: 表示矩阵行数。 如果设为0，则表示保持原有的行数不变，否则则变为设置的行数。
		
		dst2.convertTo(dst2, CV_32F);
		cv::hconcat(dst2, feature, feature2);
		float reponse=ann->predict(feature2,out);//reponse返回最大分数
		double minVal; double maxVal; Point minLoc; Point maxLoc;
		minMaxLoc(out, &minVal, &maxVal, &minLoc, &maxLoc);
		if (maxVal > 0.7)
			candidates2.push_back(candidates[i]);
	}

	// 车牌区域显示
	for (int i = 0; i < candidates2.size(); ++i)
	{
		string image = "re" + cv::format("%.4d", i);
		namedWindow(image, 0);
		cv::imshow(image, srcImage(candidates2[i]));
	}
	//之后利用矩形框的距离，位置判断车牌

	cv::waitKey(0);
	return 0;
}

