#include <iostream>
//#include <boost/filesystem.hpp>

#include "face_extraction.hpp"
#include "training_generator.hpp"

using namespace cv;
//using namespace boost::filesystem;

std::vector<std::vector<std::string>> trainingFiles;
std::vector<std::string> trainingFolders;

/*struct recursive_directory_range {
  typedef recursive_directory_range iterator;
  recursive_directory_range(path p) : p_(p) {}

  iterator begin() {
    return recursive_directory_iteratory(p_);
  }
  iterator end() {
    return recursive_directory_iterator();
  }

  path _p;
}

bool loop_through_path(const path &dir_path) {
  if(!exists(dir_path)) return false;
  for(auto it : recursive_directory_range(dir_path)) {
    std::cout << it << std::endl;
  }
} */

int main(int argc, char **argv) {
  CommandLineParser parser(argc, argv,
    "{help h||}"
    "{face_cascade|/home/naif/Documents/opencv/data/haarcascades/haarcascade_frontalface_alt.xml|}"
    "{infer|false| Infer the input}"
    "{train |false| Train the classifier}"
    "{align | false | Align images}"
    "{@image  |ferrel.jpg| image to process}");

    if(!parser.has("@image")) return -1;
    cv::Mat frame = imread(parser.get<String>(0));
    if(frame.empty()) {
      printf("No frame \n");
      return -1;
    }

    bool infer = parser.get<bool>("infer");
    bool train = parser.get<bool>("train");
    bool align = parser.get<bool>("align");

    /*path p1{"."};
    loop_through_path(p1);*/
    /* Step 2 : case 1: Training the classifier */
    if(train) {
      if(align) {
        /* Step 1 : Generating an aligned face */
        FaceExtracted faceModule = FaceExtracted(frame);
        faceModule.detectFaces();
        /* Generating landmarks */
        /* Possible to optimize */
        faceModule.generateLandmark();
        /* Aligning faces */
        faceModule.getRotatedFaces();
        /* Saving aligned faces */
        faceModule.generateThumbnails(96);
      }
      /* Step2.1 : Data augmentation */
      TrainingGenerator dataAug = TrainingGenerator("./pictures/test",frame);
      dataAug.rotateImage(50,1);
      dataAug.displayResult();
      /*Step 2.2 : Training */
    }

    /* Step 2: case 1: Infering the person */
    if(infer) {
      if(align) {
        /* Step 1 : Generating an aligned face */
        FaceExtracted faceModule = FaceExtracted(frame);
        faceModule.detectFaces();
        /* Generating landmarks */
        /* Possible to optimize */
        faceModule.generateLandmark();
        /* Aligning faces */
        faceModule.getRotatedFaces();
        /* Saving aligned faces */
        faceModule.generateThumbnails(96);
      }
      /* Step 2.1: Running python script */

      /* Step 2.2: Parsing python stdout
      Output will only contain the float percentage */

      /*Comparing to a threshold */
    }



    return 0;
}
