#include <iostream>
#include <cstdlib> /* For system calls */

#include "face_extraction.hpp"
#include "training_generator.hpp"
#include "utils.hpp"

#define NUM_THREADS 4

using namespace cv;
using namespace std;

int main(int argc, char **argv) {
  CommandLineParser parser(argc, argv,
    "{help h||}"
    "{infer | false | Infer the input}"
    "{train | false | Train the classifier}"
    "{align | false | Align images}"
    "{align_folder_in |./training-images/ | Folder containing images to align }"
    "{align_folder_out | ./aligned_images/ | Folder to contain aligned images }"
    "{@image  |ferrel.jpg| image to process}");

    parser.about("SqueezeCNN v0.0.1");
    if(parser.has("help")) {
      parser.printMessage();
    }

    
    cv::Mat frame = imread(parser.get<String>(0));
    if(frame.empty()) {
      printf("No frame \n");
    }

    string image = parser.get<string>(0);
    bool infer = parser.get<bool>("infer");
    bool train = parser.get<bool>("train");
    bool align = parser.get<bool>("align");
    String align_folder_in = parser.get<String>("align_folder_in");
    String align_folder_out = parser.get<String>("align_folder_out");

    if(parser.has("@image")) {
        FaceExtracted faceModule = FaceExtracted("./test-images/ferrel.jpg","./test-images/ferrel1.jpg");
        faceModule.detectFaces();
        faceModule.generateLandmark();
        faceModule.getRotatedFaces();
        faceModule.generateThumbnails(96);
    }
    
    /* Step 2 : case 1: Training the classifier */
    if(train) {
      #ifdef VERBOSE
      cout << "Training option selected, starting..." << endl;
      #endif
      if(align) {
        #ifdef VERBOSE
        cout << "Aliging files..." << endl;
        #endif
        boost::filesystem::path dirpath(align_folder_in);
        /* Getting list of files */
        Utils utils;
        utils.listSubPath(dirpath);
        utils.generateOutputPath(utils.getFileNames(),align_folder_out);
        #ifdef EBUG
        utils.displayMap(utils.getFileNames());
        utils.displayVector(utils.getDirectories());
        cout << "--------      Output paths        ---------" << endl;
        utils.displayMap(utils.getOutputPath());
        #endif
        std::map<string,std::vector<string>> inputs = utils.getFileNames();
        
        /* Generating aligned faces using openmp for multithreading */
        #ifdef VERBOSE
        cout << "Aligment started using " << NUM_THREADS << " threads" << endl;
        #endif
        #pragma omp parallel num_threads(NUM_THREADS)
        {
          #pragma omp single
          for(auto const& x : inputs) {
            string output;
            for(auto i=x.second.begin();i!=x.second.end()-1;i++) {
              output = "";
              utils.generateOutputPath(x.first,align_folder_out,output);
              string input = x.first + '/' +*i;
              output += '/';
              output += *i;
              #ifdef EBUG
              cout << "Input ----- " << input << "-------- Output ->" << output <<endl;
              #endif
              FaceExtracted faceModule = FaceExtracted(input,output);
              faceModule.detectFaces();
              faceModule.generateLandmark();
              faceModule.getRotatedFaces();
              faceModule.generateThumbnails(96);
            }
            
          }
        }
        /* Step2.1 : Data augmentation */
        string alignedPath = "./";
        alignedPath += align_folder_out;
        alignedPath += '/';
        boost::filesystem::path alignedFolderPath(alignedPath);
        Utils dataAugUtil;
        dataAugUtil.listSubPath(alignedFolderPath);
        inputs = dataAugUtil.getFileNames();

        #pragma omp parallel num_threads(NUM_THREADS)
        {
          #pragma omp single
          for(auto const& x : inputs) {
            string output;
            
            for(int i=0;i<x.second.size()-1;i++) {
              output = x.first + '/' + x.second[i];
              cout << output << endl;
              TrainingGenerator generator = TrainingGenerator(output,output);
              //generator.rotateImage(50,1);
              generator.flipImageHorizontally();
              generator.generateGaussianNoise(10);
            }
          }
        }

        
        #ifdef VERBOSE
        cout << "Aligment done." << endl;
        cout << "Generating embeddings in folder generated-embeddings..." << endl;
        #endif
        /* Generating embeddings */
        system("./openface/batch-represent/main.lua -outDir ./generated-embeddings/ -data ./aligned-images/");
        #ifdef VERBOSE
        cout << "Command done. Check previous log if errors occured" << endl;
        #endif

        /* Training the network */
        system("./openface/demos/classifier.py train ./generated-embeddings/");
      }
      
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
