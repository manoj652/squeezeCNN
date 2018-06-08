#include <iostream>
#include <opencv2/opencv.hpp>


class Utils {
public:
  Utils();
  float angleBetween(const cv::Point pt1,const cv::Point pt2);
  void dlibRectangleToCv(dlib::rectangle, cv::Rect*);
  void cvRectangleToDlib(cv::Rect,dlib::rectangle*);
  
};
