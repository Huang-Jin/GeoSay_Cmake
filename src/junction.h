#include <opencv2/opencv.hpp>
#include <vector>

struct Branch
{
	float branch;
	float branchScale;
	cv::Point2f endpoint;
};

typedef struct LJunction
{
	double logNFA;
	float mscale;
	float theta;
	cv::Point2f location;
	cv::Point2f center;
	size_t junctionClass;
	std::vector<Branch> branch;
	std::vector<int> neighbors;
}Junction;