//success
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;

#include <iostream>
using namespace std;

int main(int argc, char **argv)
{
	CommandLineParser parser(argc, argv,
		"{ h help           | false | print this help message }"
		"{ p proto          |    det1.prototxt   | (required) model configuration, e.g. hand/pose.prototxt }"
		"{ m model          |   det1.caffemodel    | (required) model weights, e.g. hand/pose_iter_102000.caffemodel }"
		"{ i image          |   face.jpg    | (required) path to image file (containing a single person, or hand) }"
		"{ width            |  12  | Preprocess input image by resizing to a specific width. }"
		"{ height           |  12  | Preprocess input image by resizing to a specific height. }"
		"{ t threshold      |  0.1  | threshold or confidence value for the heatmap }"
		);

	String modelTxt = parser.get<string>("proto");
	String modelBin = parser.get<string>("model");
	String imageFile = parser.get<String>("image");
	int W_in = parser.get<int>("width");
	int H_in = parser.get<int>("height");
	float thresh = parser.get<float>("threshold");
	if (parser.get<bool>("help") || modelTxt.empty() || modelBin.empty() || imageFile.empty())
	{
		cout << "A sample app to demonstrate human or hand pose detection with a pretrained OpenPose dnn." << endl;
		parser.printMessage();
		return 0;
	}

	// read the network model
	Net net = readNetFromCaffe(modelTxt, modelBin);

	// and the image
	Mat img = imread(imageFile);
	if (img.empty())
	{
		std::cerr << "Can't read image from the file: " << imageFile << std::endl;
		exit(-1);
	}

	// send it through the network
	Mat inputBlob = blobFromImage(img, 1.0, Size(W_in, H_in), Scalar(0, 0, 0), false, false);
	net.setInput(inputBlob,"data");
	Mat result = net.forward("prob1");
	
	// the result is an array of "heatmaps", the probability of a body part being in location x,y

	Mat probMat = result.reshape(1, 1); //输出连个标签，第一个代表人脸，第二个代表非人脸，经验证：输入face照片，输出[1,0];输入非人脸照片，输出[0,1]
	cout << probMat.type() << endl;
	float a = probMat.at<float>(0, 0);//人脸的得分
	float b = probMat.at<float>(0, 1);//非人脸的得分
	namedWindow("facedetect", 0);
	imshow("facedetect", probMat);
	waitKey();

	return 0;
}
