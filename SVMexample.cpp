#include<iostream>
#include<opencv2\opencv.hpp>
#include<string>
#include<opencv2\ml\ml.hpp>

using namespace cv;
using namespace std;

void convert_to_ml(const vector< Mat > & train_samples, Mat& trainData)
{
	//--Convert data
	const int rows = (int)train_samples.size();
	const int cols = (int)train_samples[0].cols*train_samples[0].rows;
	trainData = Mat(rows, cols, CV_32FC1);//一定要类型转换32FC1
	for (int i = 0; i < train_samples.size(); ++i)
	{
		Mat p = train_samples[i].reshape(1, 1);//一行向量
		p.copyTo(trainData.row(i));//trainData.row(i)返回Mat类型
	}
}
int main() {
	string file;
	vector<string> files;
	vector<String> nofiles,testnofiles;
	vector<Mat> image, testimage;
	vector<int> labels;
	string fileName; 
	Mat srcImage,dst;
	for (int i = 1; i < 601; i++)   //600
	{
		stringstream ss;
		string str;
		ss << 1;
		ss >> str;
		fileName = "E:\\test_opencv\\EasyPR-master\\resources\\train\\svm\\has\\train\\plate_" + str + ".jpg";
		files.push_back(fileName);
		srcImage = imread(fileName);
		cvtColor(srcImage, dst, CV_RGB2GRAY);
		resize(dst, dst, Size(120, 30)); //30*120
		image.push_back(dst);
		labels.push_back(1);
	}
	fileName = "E:\\test_opencv\\EasyPR-master\\resources\\train\\svm\\no\\train\\*.jpg";
	glob(fileName,nofiles,false);
	for (int i = 0; i < 600; i++) {//nofiles.size()
		Mat src = imread(nofiles[i]);
		cvtColor(src, dst, CV_RGB2GRAY);//36*136
		resize(dst,dst,Size(120,30)); //30*120
		image.push_back(dst); 
		labels.push_back(0);
	}
	Mat traindata,testdata;
	convert_to_ml(image, traindata);
	Mat label = cv::Mat(labels);

	
	Ptr<ml::SVM> svm = ml::SVM::create();
	svm->setCoef0(0.0);
	svm->setKernel(ml::SVM::LINEAR);//RBF
	svm->setType(ml::SVM::C_SVC);
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
	svm->train(traindata, 0, label);
	cout << "train final" << endl;
	Mat supportvector=svm->getSupportVectors();

//////////////////////////////test//////////////////////////////////

	//单个样本预测
	Mat sample = traindata.row(3);
	float repon=svm->predict(sample);
	
	//多个样本预测
	Mat classa;
	Mat row = traindata.rowRange(1, 50).clone();
	svm->predict(row, classa);
	vector<float> re;
	for (int i = 0; i < 49; i++) {
		float a = classa.at<float>(i, 0);
		re.push_back(a);   //证明我们的预测器是可以正常工作的，只是追确率堪忧！！
	}

	//单个测试样本
	Mat src = imread("E:\\test_opencv\\EasyPR-master\\resources\\train\\svm\\has\\train\\plate_36.jpg");
	cvtColor(src, dst, CV_RGB2GRAY);//36*136
	resize(dst, dst, Size(120, 30));
	Mat s = dst.reshape(1, 1);
	s.convertTo(s, CV_32FC1);
	float r = svm->predict(s);

	//多个测试样本预测
	fileName = "E:\\test_opencv\\EasyPR-master\\resources\\train\\svm\\has\\test\\*.jpg";
	glob(fileName, testnofiles, false);
	for (int i = 0; i < 100; i++) {//nofiles.size()
		Mat src = imread(testnofiles[i]);
		cvtColor(src, dst, CV_RGB2GRAY);//36*136
		resize(dst, dst, Size(120, 30)); //30*120
		testimage.push_back(dst);
	}
	convert_to_ml(testimage, testdata);
	Mat class1;
	svm->predict(testdata, class1);
	vector<float> reponse;
	for (int i = 0; i < 100; i++) {
		float a = class1.at<float>(i, 0);
		reponse.push_back(a);
	}
 	waitKey(0);

}