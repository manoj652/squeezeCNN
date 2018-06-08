#ifndef TRAINING_GENERATOR_SQUEEZE
#define TRAINING_GENERATOR_SQUEEZE

#include <iostream>
#include <opencv2/opencv.hpp>

/*
* Class allowing to perform data augmentation in order to multiply
* our original set of images.
*/
class TrainingGenerator {
private:
  cv::Mat _originalFrame;
  cv::Mat _alteredFrame;
  std::vector<cv::Mat> listOfAlteredFrames;
public:
  TrainingGenerator(std::string pathToImage);
  TrainingGenerator(cv::Mat matrix);

  void rotateImage(double,int);
  void flipImage();
  void generateGaussianNoise();
  void saveMatrix(std::string);
  void saveMatrix(std::string,cv::Mat);
  /* optional */
  void generateConditionalGan();
};

#endif //TRAINING_GENERATOR_SQUEEZE
