#include <iostream>

#include "face_extraction.hpp"

using namespace cv;

int main(int argc, char **argv) {
  CommandLineParser parser(argc, argv,
    "{help h||}"
    "{face_cascade|/home/naif/Documents/opencv/data/haarcascades/haarcascade_frontalface_alt.xml|}"
    "{@image  |ferrel.jpg| image to process}");

    if(!parser.has("@image")) return -1;
    cv::Mat frame = imread(parser.get<String>(0));
    if(frame.empty()) {
      printf("No frame \n");
      return -1;
    }

    FaceExtracted faceModule = FaceExtracted(frame);
    faceModule.detectFaces();
    faceModule.generateLandmark();
    faceModule.displayResult(2);
    faceModule.saveCroppedFaces("pictures");

    return 0;
}
