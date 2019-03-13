//LBP检测器的运用
//本程序可以运行，用于人脸识别和人眼识别

#include<opencv2/opencv.hpp>
#include <iostream>   
using namespace std;
using namespace cv;

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
void detectAndDisplay(Mat frame);

int main(int argc, char** argv)
{
	Mat srcImage;
	srcImage = imread("1.jpg", 1);    
	imshow("原图", srcImage);
	//============加载分类器=========
  
	if (!face_cascade.load("D:\\Program Files\\OpenCV\\opencv\\sources\\data\\lbpcascades\\lbpcascade_frontalface.xml"))//也可用Haar分类器
	{
		printf("人脸检测器加载失败\n");
		return -1;
	}
	if (!eyes_cascade.load("D:\\Program Files\\OpenCV\\opencv\\sources\\data\\haarcascades_cuda\\haarcascade_eye.xml"))
	{
		printf("人眼检测器加载失败\n");
		return -1;
	};
	//============调用人脸检测函数  =========
	detectAndDisplay(srcImage);
	waitKey(0); 
}


void detectAndDisplay(Mat dispFace)
{
	//定义变量
	std::vector<Rect> faces;
	std::vector<Rect>eyes;
	Mat srcFace, grayFace, eqlHistFace;
	int eye_number = 0;

	cvtColor(dispFace, grayFace, CV_BGR2GRAY);   
	equalizeHist(grayFace, eqlHistFace);   //直方图均衡化  
    //人脸检测******************
	face_cascade.detectMultiScale(eqlHistFace, faces, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10));
	//增大第四个参数可以提高检测精度,但也可能会造成遗漏
	//人脸尺寸minSize和maxSize，关键参数，自行设定，随图片尺寸有很大关系，

	for (unsigned int i = 0; i < faces.size(); i++)
	{
		//用绿色椭圆标记检测到的人脸
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(dispFace, center, Size(faces[i].width / 2, faces[i].height * 65 / 100), 0, 0, 360, Scalar(255, 0, 0), 2, 8, 0);
		//人眼检测*****************
		Mat faceROI = eqlHistFace(faces[i]);
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.2, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(15, 15), Size(80, 80));
		eye_number += eyes.size();//人眼计数

		 //用绿色圆标记检测到的人眼*****************
		for (unsigned int j = 0; j <eyes.size(); j++)
		{
			Point center(faces[i].x + eyes[j].x + eyes[j].width / 2, faces[i].y + eyes[j].y + eyes[j].height / 2);
			int radius = cvRound((eyes[j].width + eyes[i].height)*0.25);
			circle(dispFace, center, radius, Scalar(105, 50, 255), 2, 8, 0);
		}
	}
	//*****************3.0检测结果输出*****************
	cout << "检测结果\n人脸: " << faces.size() << " 张" << endl;
	cout << "人眼: " << eye_number << " 只" << endl;
	imshow("人脸识别结果", dispFace);
}
