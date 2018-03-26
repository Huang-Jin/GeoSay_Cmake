#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <fstream>
#include<numeric>

#include "junction.h"
#include "prob.h"

using namespace cv;
using namespace std;

#define PI 3.141592653

typedef struct Parameter
{
	string asjPath;
	bool useBTHat;
}Parameter;

typedef vector<vector<float>> DISMAT;

inline float diff_circular(const float a, const float b)
{
	float temp = abs(a - b);
	return (MIN(temp, 2*PI - temp));
}

void readASJ(string filename, vector<Junction> & junctions)
{
	int n;
	fstream _file;
	_file.open(filename, ios::in);
	_file >> n;

	if (!junctions.empty())
	{
		junctions.clear();
	}
	
	for (int i = 0; i < n; ++i)
	{
		Junction junct;
		Branch branch;
		int junctionClass, scale, r_d;
		float logNFA, branchstrength;
		
		_file >> junct.location.x >> junct.location.y;
		_file >> junctionClass >> scale >> r_d >> logNFA;
		junct.junctionClass = (size_t)junctionClass;
		junct.logNFA = logNFA;

		for (int j = 0; j < junct.junctionClass; ++j)
		{
			_file >> branch.branch >> branchstrength >> branch.branchScale;
			branch.endpoint = junct.location + cv::Point2f(branch.branchScale*cos(branch.branch), branch.branchScale*sin(branch.branch));
			junct.branch.push_back(branch);
		}

		junct.mscale = junct.branch[0].branchScale;
		for (int j = 1; j < junct.junctionClass; ++j)
		{
			if (junct.branch[j].branchScale < junct.mscale)
			{
				junct.mscale = junct.branch[j].branchScale;
			}
		}

		junct.theta = diff_circular(junct.branch[0].branch, junct.branch[1].branch);

		junctions.push_back(junct);
	}
	_file.close();
}

vector<Junction> getJunctions(string filename, Parameter param)
{
	vector<Junction> junctions;
	string asjFile = filename + ".asj";

	fstream _file;
	_file.open(asjFile, ios::in);

	if (!_file)		// ASJ file is not detected
	{
		string command = param.asjPath + " " + filename;
		system(command.c_str());
	}
	_file.close();

	// ASJ file has been detected, read it to memory
	readASJ(asjFile, junctions);

	return junctions;
}

void calcCenter(vector<Junction> & junctions)
{
	for (int i = 0; i < junctions.size(); i++)
	{
		Point2f pt(0, 0);
		for (int j = 0; j < junctions[i].junctionClass; j++)
		{
			pt += junctions[i].branch[j].endpoint;
		}
		junctions[i].center.x = pt.x / junctions[i].junctionClass;
		junctions[i].center.y = pt.y / junctions[i].junctionClass;
	}
}

inline float norm2(Point2f p1, Point2f p2)
{
	return (p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y);
}

void calcDisMat(vector<Junction> & junctions, DISMAT & disMat)
{
	int len = junctions.size();

	for (int i = 0; i < len; i++)
	{
		vector<float> dis;
		for (int j = 0; j < len; j++)
		{
			dis.push_back(norm2(junctions[i].center, junctions[j].center));
		}
		disMat.push_back(dis);
	}
}

bool similiarScale(Junction & j1, Junction & j2)
{
	const float thresh = 3;
	float ratio = j1.mscale / j2.mscale;
	return (ratio > thresh || ratio < 1 / thresh);
}

// Sort template which returns the indexes
template <typename T>
vector<size_t> sort_indexes(const vector<T> &v) {

	// 初始化索引向量
	vector<size_t> idx(v.size());
	//使用iota对向量赋0~？的连续值
	iota(idx.begin(), idx.end(), 0);

	// 通过比较v的值对索引idx进行升序排序
	sort(idx.begin(), idx.end(),
		[&v](size_t i1, size_t i2) {return v[i1] < v[i2]; });	// 降序 [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
	return idx;
}

void findNeighbors(vector<Junction> & junctions, DISMAT & disMat)
{
	const int k = 6;
	for (int i = 0; i < junctions.size(); i++)
	{
		vector<float> dis;
		dis = disMat.at(i);
		float thresh = junctions[i].mscale;

		vector<size_t> indexes = sort_indexes(dis);
		junctions[i].neighbors.clear();
		for (int j = 0; j < indexes.size(); j++)
		{
			if (dis[indexes[j]] > thresh*thresh || junctions[i].neighbors.size() >= k)
				break;

			if (!similiarScale(junctions[i], junctions[indexes[j]]))
			{
				junctions[i].neighbors.push_back(indexes[j]);
			}
		}
	}
}

void getMask(Junction junc, Mat & mask, Mat & im)
{
	mask.create(Size(im.cols, im.rows), CV_8UC1);
	mask.setTo(0);

	Point pts[1][4];
	pts[0][0] = junc.branch[0].endpoint;
	pts[0][1] = junc.location;
	pts[0][2] = junc.branch[1].endpoint; 
	pts[0][3] = junc.branch[0].endpoint - junc.location + junc.branch[1].endpoint;
	
	const Point* ppt[1] = { pts[0] };
	int npt[] = { 4 };
	fillPoly(mask, ppt, npt, 1, Scalar(255));
}

void calcGBI(Mat & gbi, vector<Junction> & junctions, DISMAT & disMat, bool useBTHat = true)
{
	gbi.setTo(0);
	Mat temp;
	gbi.copyTo(temp);
	for (int i = 0; i < junctions.size(); i++)
	{
		temp.setTo(0);
		int indi = floor(junctions[i].theta / PI * 1000);
		float sa = 0;
		float w = g_prob[indi] * (1 - exp(junctions[i].logNFA));

		for (int j = 1; j < junctions[i].neighbors.size();j++)
		{
			int index = junctions[i].neighbors[j];
			int indj = floor(junctions[index].theta / PI * 1000);

			w += exp(-sqrt(disMat[i][index]/junctions[i].mscale))
				*g_prob[indj] * (1 - exp(junctions[index].logNFA));
		}

		Mat mask;
		getMask(junctions[i], mask, gbi);
		temp.setTo(w, mask);
		printf("%.2f\r\n", w);
		gbi += temp;
	}
}

void getLumination(Mat & image, Mat & lum)
{
	if (image.channels() == 1)
	{
		image.copyTo(lum);
	}
	else
	{
		//int ch = image.channels();
		lum.create(Size(image.cols, image.rows), CV_32FC1);
		for (int i = 0; i < image.rows; i++)
		{
			for (int j = 0; j < image.cols; j++)
			{
				//int depth = image.depth();
				int max = 0;
				for (int k = 0; k < image.channels(); k++)
				{
					max = image.at<Vec3b>(i, j)[k] > max ? image.at<Vec3b>(i, j)[k] : max;
				}
				lum.at<float>(i, j) = max;
			}
		}
	}
}

Mat Geosay(string filename, Parameter param)
{
	vector<Junction> junctions;
	junctions = getJunctions(filename, param);
	calcCenter(junctions);

	DISMAT disMat;
	calcDisMat(junctions, disMat);
	findNeighbors(junctions, disMat);

	Mat image = imread(filename);
	Mat ret;
	ret.create(Size(image.cols, image.rows), CV_32FC1);
	ret.setTo(0);
	calcGBI(ret, junctions, disMat, param.useBTHat);

	//Mat element = getStructuringElement(MORPH_RECT, Size(50, 50));
	//imshow("element", element*128);
	//waitKey(0);

	Mat lum;
	getLumination(image, lum);
	//printf("%.4f", lum.at<float>(10, 9));

	Mat temp;
	if (param.useBTHat)
	{
		morphologyEx(lum, temp, MORPH_BLACKHAT, getStructuringElement(MORPH_RECT, Size(50, 50)));
		//printf("%.4f", temp.at<float>(10, 9));
		ret = ret.mul(1-temp/255);
	}

	//printf("%.4f", ret.at<float>(10, 9));
	GaussianBlur(ret, ret, Size(5, 5), 1);

	//printf("%.4f", ret.at<float>(10, 9));

	return ret;
}

int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		printf("Parameter error! please enter the paths of ASJDetector and image.");
	}

	Parameter param;
	param.asjPath = argv[1];	// Can not use form like this ../3rdparty/asj/ASJDetector.exe
	param.useBTHat = true;

	string filename = argv[2];

	Mat result = Geosay(filename, param);

	imwrite("gbi.png", result);
	system("pause");

	return 0;
}

