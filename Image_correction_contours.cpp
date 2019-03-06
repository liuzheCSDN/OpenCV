#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
	Mat srcImage = imread("D:\\Program Files\\OpenCV\\opencv\\sources\\samples\\data\\imageTextR.png");

	if (srcImage.cols>1000 || srcImage.rows>800) {//图片过大，进行降采样
		pyrDown(srcImage, srcImage);
		pyrDown(srcImage, srcImage);
		pyrDown(srcImage, srcImage);
	}

	Mat grayImage, binaryImage;
	cvtColor(srcImage, grayImage, CV_BGR2GRAY);//转化灰度图
	adaptiveThreshold(grayImage, binaryImage, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, 0);//自适应滤波

	vector<vector<Point> > contours;
	//RETR_EXTERNAL:表示只检测最外层轮廓
	//CHAIN_APPROX_NONE：获取每个轮廓的每个像素，相邻的两个点的像素位置差不超过1 
	findContours(binaryImage, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	//获得矩形包围框,之所以先用boundRect，是为了使用它的area方法求面积，而RotatedRect类不具备该方法
	float area = boundingRect(contours[0]).area(); 
	int index = 0;
	for (int i = 1; i<contours.size(); i++)
	{

		if (boundingRect(contours[i]).area()>area)
		{
			area = boundingRect(contours[i]).area();
			index = i;
		}
	}
	Rect maxRect = boundingRect(contours[index]);//找出最大的那个矩形框（即最大轮廓）
	Mat ROI = binaryImage(maxRect);
	imshow("maxROI", ROI);

	RotatedRect rect = minAreaRect(contours[index]);//获取对应的最小矩形框，这个长方形是倾斜的
	Point2f rectPoint[4];
	rect.points(rectPoint);//获取四个顶点坐标，这是RotatedRect类定义的方法
	double angle = rect.angle;
	//angle += 90;
	Point2f center = rect.center;


	drawContours(binaryImage, contours, -1, Scalar(255), CV_FILLED);
	// srcImage.copyTo(RoiSrcImg,binaryImage);
	Mat RoiSrcImg = Mat::zeros(srcImage.size(), srcImage.type());
	srcImage.copyTo(RoiSrcImg);

	Mat Matrix = getRotationMatrix2D(center, angle, 0.8);//得到旋转矩阵算子，0.8缩放因子
	warpAffine(RoiSrcImg, RoiSrcImg, Matrix, RoiSrcImg.size(), 1, 0, Scalar(0, 0, 0));

	imshow("src Image", srcImage);
	imshow("contours", binaryImage);
	imshow("recorrected", RoiSrcImg);

	while (waitKey() != 'q') {}
	return 0;
}