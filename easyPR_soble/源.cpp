#include <iostream>
#include <sstream>
#include<opencv2\highgui.hpp>
#include<opencv2\ml\ml.hpp>
#include<opencv2\opencv.hpp>
#include<time.h>

using namespace std;
using namespace cv;

bool verifySizes(RotatedRect candidate);
void GetTextProject(Mat src, vector<int>& pos);
int main()
{
	Mat input, img_gray, img_sobel, img_threshold, result;
	input = imread("16.jpg");
	cvtColor(input, img_gray, CV_BGR2GRAY);//转为灰度图像
	blur(img_gray, img_gray, Size(5, 5));  //使用5*5高斯模糊去噪

	Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0);//采用sobel滤波器找竖直边

	//二值化图像，阈值由Otsu算法得到,忽略阈值参数（threshold函数中的第三个参数）
	threshold(img_sobel, img_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
	imshow("img_threshold", img_threshold);
	Mat element = getStructuringElement(MORPH_RECT, Size(17, 3)); /*定义闭形态学算子中使用的结构元素。*/
	morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element);

	

	//findContours函数去除非车牌号区域
	vector < vector < Point > > contours;
	findContours(img_threshold, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);


	vector <vector <Point>> ::iterator itc = contours.begin();
	vector <RotatedRect> rects;

	// 根据面积和宽高比，移除非车牌矩形框
	while (itc != contours.end())
	{
		RotatedRect mr = minAreaRect(Mat(*itc));
		if (!verifySizes(mr))
			itc = contours.erase(itc);//删除vector容器中的一个或者一段元素，删除一个元素的时候，其参数为指向相应元素的迭代器
		else
		{
			++itc;
			rects.push_back(mr);//满足条件的矩形框存入
		}
	}
	printf("检测到的车牌候选区域个数：%d",rects.size());
	/*for (int i = 0; i < rects.size(); i++) {
		Point2f point[4];
		rects[i].points(point);//外接矩形的4个顶点
		for (int i = 0; i < 4; i++)//绘制外接矩形
			line(input, point[i], point[(i + 1) % 4], Scalar(255, 0, 0));
	}
	imshow("input", input);*/

	input.copyTo(result);
	/*利用漫水填充算法来得到更精确的矩形*/
	for (int i = 0; i < rects.size()-1; i++)
	{
		circle(result, rects[i].center, 3, Scalar(0, 255, 0), -1);
		imshow("result", result);
		//Get the min size between width and height
		float minSize = (rects[i].size.width < rects[i].size.height) ? rects[i].size.width : rects[i].size.height;
		minSize = minSize * 0.5;
		srand(time(NULL));//表示设置一个随机种子,每次运行都能保证随机种子不同

		//初始化漫水填充的参数
		Mat mask;
		mask.create(input.rows + 2, input.cols + 2, CV_8UC1);
		mask = Scalar::all(0);
		int loDiff = 30;
		int upDiff = 30;
		int connectivity = 4;
		int newMaskVal = 255;
		int NumSeeds = 10;
		Rect ccomp;
		int flags = connectivity + (newMaskVal << 8) + CV_FLOODFILL_FIXED_RANGE + CV_FLOODFILL_MASK_ONLY;
		for (int j = 0; j < NumSeeds; j++)
		{
			Point seed;
			seed.x = rects[i].center.x + rand() % (int)minSize - (minSize / 2);
			seed.y = rects[i].center.y + rand() % (int)minSize - (minSize / 2);
			circle(result, seed, 1, Scalar(0, 255, 255), -1);
			int area = floodFill(input, mask, seed, Scalar(255, 0, 0), &ccomp, Scalar(loDiff, loDiff, loDiff), Scalar(upDiff, upDiff, upDiff), flags);
		}

		//imshow("mask", mask);

		//Check new floodfill mask match for a correct patch.
		//Get all points detected for minimal rotated Rect.
		vector <Point> pointsInterest;
		Mat_<uchar> ::iterator itmask = mask.begin<uchar>();
		Mat_<uchar>::iterator end = mask.end<uchar>();
		for (; itmask != end; ++itmask)
			if (*itmask == 255)
				pointsInterest.push_back(itmask.pos());//pos()返回坐标点
		RotatedRect minRect = minAreaRect(pointsInterest);
		bool a = verifySizes(minRect);
		if (a)
		{
			// rotated rectangle drawing 
			Point2f rect_points[4]; minRect.points(rect_points);
			for (int j = 0; j < 4; j++)
				line(result, rect_points[j], rect_points[(j + 1) % 4], Scalar(0, 0, 255), 1, 8);
			imshow("result", result);
			//Get rotation matrix
			float r = (float)minRect.size.width / (float)minRect.size.height;
			float angle = minRect.angle;
			if (r < 1)
				angle = 90 + angle;
			Mat rotmat = getRotationMatrix2D(minRect.center, angle, 1);

            
			//Create and rotate image.
			Mat img_rotated;
			warpAffine(input, img_rotated, rotmat, input.size(), CV_INTER_CUBIC);  //仿射变换
			
			Size rect_size = minRect.size;
			if (r < 1)
				swap(rect_size.width, rect_size.height);//加它是为了调整使width大于height
			Mat img_crop;
			getRectSubPix(img_rotated, rect_size, minRect.center, img_crop);//从原图像中提取提取一个感兴趣的矩形区域图像
			//namedWindow("img_crop", CV_WINDOW_NORMAL);
			//imshow("img_crop", img_crop);

			Mat resultResized;
			resultResized.create(36, 136, CV_8UC3);//尺寸变为36*136
			resize(img_crop, resultResized, resultResized.size(), 0, 0, INTER_CUBIC);
			Mat grayResult;
			cvtColor(resultResized, grayResult, CV_BGR2GRAY);
			imshow("grayResult", grayResult);

			//equalizeHist(grayResult, grayResult);//加了以后效果反而不好
			//imshow("gray3Result", grayResult);
			Mat Result_threshold;
			threshold(grayResult, Result_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
			imshow("gray4Result", Result_threshold);
			vector<int> pos(172, 0);

			GetTextProject(Result_threshold, pos);


			//分类车牌号与非车牌号
			Ptr<ml::SVM> svm = ml::SVM::create();
			svm->load("svm_hist.xml");
			Mat p = Mat(pos, true);
			p.convertTo(p, CV_32FC1);
			int pred = (int)svm->predict(p); //返回标签
			printf("%d\n", pred);

		}

	} 


	waitKey(0);
	system("pause");
	return 0;
}

//对检测到的区域做一些基本的验证，这些验证基于面积与宽高比。
//车牌区域：有40%的误差范围内，其宽高比为520/110 = 4.727272 

bool verifySizes(RotatedRect candidate)
{
	float error = 0.4;
	//Spain car plate size: 52*11 aspect 4,7272
	const float aspect = 4.7272;
	//Set a min and max area, All other patches are discared
	int min = 15 * aspect * 15;
	int max = 125 * aspect * 125;
	//Get only patches that match to a respect ratio.
	float rmin = aspect - aspect * error;
	float rmax = aspect + aspect* error;

	int area = candidate.size.height * candidate.size.width;
	float r = (float)candidate.size.width / (float)candidate.size.height;
	if (r  < 1)
		r = 1 / r;
	if ((area < min || area > max) || (r < rmin || r > rmax))
		return false;
	else
		return true;
}


void GetTextProject(Mat src, vector<int>& pos) {// pos就是用于存储垂直投影和水平投影的位置

		
	for (int i = 0; i < src.cols; i++)// H_PROJECT
	{
		for (int j = 0; j < src.rows; j++) 
		{
			if (src.at<uchar>(j, i) == 255)
				pos[j]++;//36
		}
	}
		for (int i = 0; i < src.rows; i++)//v_PROJECT
			for (int j = 0; j < src.cols; j++)//136
				if (src.at<uchar>(i, j) == 255)
					pos[j+36]++;
		
}