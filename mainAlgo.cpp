#include <iostream>

#include "face_extraction.hpp"

using namespace cv;

int main(int argc, char **argv) {
  CommandLineParser parser(argc, argv,
    "{help h||}"
    "{face_cascade|/home/naif/Documents/opencv/data/haarcascades/haarcascade_frontalface_alt.xml|}"
    "{infer|false| Infer the input}"
    "{train |false| Train the classifier}"
    "{@image  |ferrel.jpg| image to process}");

    if(!parser.has("@image")) return -1;
    cv::Mat frame = imread(parser.get<String>(0));
    if(frame.empty()) {
      printf("No frame \n");
      return -1;
    }

    bool infer = parser.get<bool>("infer");
    bool train = parser.get<bool>("train");
    /* Step 1 : Generating an aligned face */
    FaceExtracted faceModule = FaceExtracted(frame);
    faceModule.detectFaces();
    /* Generating landmarks */
    /* Possible to optimize */
    faceModule.generateLandmark();
    /* ALigning faces */
    faceModule.getRotatedFaces();
    /* Saving aligned faces */
    faceModule.generateThumbnails(96);

    /* Step 2 : case 1: Training the classifier */
    if(train) {
      /* STep2.1 : Data augmentation */
      /*Step 2.2 : Training */
    }

    /* Step 2: case 1: Infering the person */
    if(infer) {
      /* Step 2.1: Running python script */
      /* Step 2.2: Parsing python stdout
      Output will only contain the float percentage */

      /*Comparing to a threshold */
    }



    return 0;
}
