#include <iostream>
/* Opencv */
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>


#define PADDING 15 /* 15 Extra pixels to be sure to get the face */



class FaceExtracted {
private:
  cv::CascadeClassifier _faceCascade;
  std::string face_cascade_name;
  
  std::string window_name;
  std::vector<cv::Mat> facesROI;
  cv::Mat motherFrame;
  std::vector<cv::Rect> _faces;
  std::vector<cv::Rect> _eyes;
  std::vector<std::vector<cv::Point2f>> _landmarks;
  void drawPolyline(const int,const int, const int,bool);
public:

  FaceExtracted(cv::Mat);

  void detectFaces();
  void alignDetectedFace();
  void saveCroppedFaces(std::string path);
  int generateLandmark();
  void displayResult(int);
};
