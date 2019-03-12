//https://blog.csdn.net/qq_30815237/article/details/87914775
//opencv基于DNN的人脸检测
#include <iostream>
#include <cstdlib>
#include<opencv2\dnn\dnn.hpp>
#include<opencv2\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
using namespace cv;
using namespace cv::dnn;
using namespace std;
const size_t inWidth = 300; //output image size
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const Scalar meanVal(104.0, 117.0, 123.0);

int main(int argc, char** argv)
{
	float min_confidence = 0.5;
	String modelConfiguration = "D:\\Program Files\\OpenCV\\opencv\\sources\\samples\\dnn\\face_detector\\deploy.prototxt";
	String modelBinary = "D:\\Program Files\\OpenCV\\opencv\\sources\\samples\\dnn\\face_detector\\res10_300x300_ssd_iter_140000_fp16.caffemodel";
	//! [Initialize network]
	dnn::Net net = readNetFromCaffe(modelConfiguration, modelBinary);//Reads a network model stored in Caffe model in memory
	//! [Initialize network]
	if (net.empty())
	{
		cerr << "Can't load network by using the following files: " << endl;
		cerr << "prototxt:   " << modelConfiguration << endl;
		cerr << "caffemodel: " << modelBinary << endl;
		cerr << "Models are available here:" << endl;
		cerr << "<OPENCV_SRC_DIR>/samples/dnn/face_detector" << endl;
		cerr << "or here:" << endl;
		cerr << "https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector" << endl;
		exit(-1);
	}

	//VideoCapture cap(0);//打开摄像头
	VideoCapture cap("E:/test_opencv/facedetect/facedetect/face.avi");
	if (!cap.isOpened())
	{
		cout << "Couldn't open camera : " << endl;
		return -1;
	}
	for (;;)//死循环
	{
		Mat frame;
		cap >> frame; // get a new frame from camera/video or read image

		if (frame.empty())
		{
			waitKey();
			break;
		}

		if (frame.channels() == 4)
			cvtColor(frame, frame, COLOR_BGRA2BGR);

		//! [Prepare blob]
		Mat inputBlob = blobFromImage(frame, inScaleFactor,
			Size(inWidth, inHeight), meanVal, false, false); //Convert Mat to batch of images
															 //! [Prepare blob]

															 //! [Set input blob]
		net.setInput(inputBlob, "data"); //set the network input
										 //! [Set input blob]

										 //! [Make forward pass]
          //compute output，这是一个4D数，rows and cols can only hold 2 dimensions, so they are not used here, and set to -1
		Mat detection = net.forward("detection_out");   //! [Make forward pass]

		vector<double> layersTimings;
		double freq = getTickFrequency() / 1000;//用于返回CPU的频率。get Tick Frequency。这里的单位是秒，也就是一秒内重复的次数。
		double time = net.getPerfProfile(layersTimings) / freq;

		Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());//101*7矩阵

		ostringstream ss;
		ss << "FPS: " << 1000 / time << " ; time: " << time << " ms";
		putText(frame, ss.str(), Point(20, 20), 0, 0.5, Scalar(0, 0, 255));

		float confidenceThreshold = min_confidence;
		for (int i = 0; i < detectionMat.rows; i++)
		{
			float confidence = detectionMat.at<float>(i, 2);      //第二列存放可信度

			if (confidence > confidenceThreshold)//满足阈值条件
			{
				//存放人脸所在的图像中的位置信息
				int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
				int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
				int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
				int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

				Rect object((int)xLeftBottom, (int)yLeftBottom,//定义一个矩形区域（x,y,w,h)
					(int)(xRightTop - xLeftBottom),
					(int)(yRightTop - yLeftBottom));

				rectangle(frame, object, Scalar(0, 255, 0));//画个边框

				ss.str("");
				ss << confidence;
				String conf(ss.str());
				String label = "Face: " + conf;
				int baseLine = 0;
				Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
				rectangle(frame, Rect(Point(xLeftBottom, yLeftBottom - labelSize.height),
					Size(labelSize.width, labelSize.height + baseLine)),
					Scalar(255, 255, 255), CV_FILLED);
				putText(frame, label, Point(xLeftBottom, yLeftBottom),
					FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0));
			}
		}
		cv::imshow("detections", frame);
		if (waitKey(1) >= 0) break;
	}
	return 0;
}