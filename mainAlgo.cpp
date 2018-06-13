#include <iostream>
#include <cstdlib> /* For system calls */

#include "./src/face_extraction.hpp"
#include "./src/training_generator.hpp"
#include "./src/utils.hpp"

#define NUM_THREADS 4
#define THRESHOLD 0.7

using namespace cv;
using namespace std;

void inferFace(string pathToFrame) {
   string infercommand = "python classifier.py  infer ./generated-embeddings/classifier.pkl ";
   string pathToComm = pathToFrame;
   infercommand += pathToComm; 
   char* command = new char[infercommand.length() +1];
   std::strcpy(command,infercommand.c_str());
   FILE* pip = popen(command,"r");
   if(!pip) {
     cerr << "Could not start command" << endl;
   }

    std::array<char,128> buffer;
    string resultat;
    while(fgets(buffer.data(),128,pip) != NULL) {
      resultat += buffer.data();
    }
    Utils parser;
    std::vector<string> parserResult = parser.splitText(resultat,' ');
    float accuracy = strtof(parserResult[1].c_str(),0);            
             
     #ifdef VERBOSE
             cout << "For path : " << pathToComm << endl;
     #endif
      /*Comparing to a threshold */
     if(accuracy < THRESHOLD) {
         cout << "Result not accurate enough" << endl;
         cout << "******** Detected " << parserResult[0] << " with " << accuracy * 100 << " accuracy." << endl;
     } else cout << "Detected " << parserResult[0] <<" with " << accuracy * 100 << " % accuracy." << endl;
      cout << endl;
      cout << endl;
}

int main(int argc, char **argv) {
  CommandLineParser parser(argc, argv,
    "{help h||}"
    "{infer | false | Infer the input}"
    "{video | | Video stream }"
    "{train | false | Train the classifier}"
    "{align | false | Align images}"
    "{align_folder_in |./training-images/ | Folder containing images to align }"
    "{align_folder_out | ./aligned_images/ | Folder to contain aligned images }"
    "{@image  || image to process}");

    parser.about("SqueezeCNN v0.0.1");
    if(parser.has("help")) {
      parser.printMessage();
    }


    string image = parser.get<string>(0);
    string video = parser.get<string>("video");
    bool infer = parser.get<bool>("infer");
    bool train = parser.get<bool>("train");
    bool align = parser.get<bool>("align");
    String align_folder_in = parser.get<String>("align_folder_in");
    String align_folder_out = parser.get<String>("align_folder_out");

    /*if(parser.has("@image")) {
        FaceExtracted faceModule = FaceExtracted("./test-images/ferrel.jpg","./test-images/ferrel1.jpg");
        faceModule.detectFaces();
        faceModule.generateLandmark();
        faceModule.getRotatedFaces();
        faceModule.generateThumbnails(96);
    }*/
    std::vector<string> outputAlignement;
    if(align) {
      #ifdef VERBOSE
        cout << "Aliging files..." << endl;
      #endif
      boost::filesystem::path dirpath(align_folder_in);
      /* Recuperating list of files into a map */
      Utils utils;
      utils.listSubPath(dirpath);
      utils.generateOutputPath(utils.getFileNames(),align_folder_out);
      #ifdef EBUG
        cout << " ----------- Displaying input folder tree ------------ " << endl;
        utils.displayMap(utils.getFileNames());
        cout << " ----------- Displaying predicted output tree ------------ "<< endl;
        utils.displayMap(utils.getOutputPath());
      #endif
      std::map<string,std::vector<string>> inputs = utils.getFileNames();

      /* Generating aligned faces in output folder */
      #ifdef VERBOSE
        cout << "Aligment started using " << NUM_THREADS << " threads" << endl;
      #endif
      
      #pragma omp parallel num_threads(NUM_THREADS)
      {
        #pragma omp single
        for(auto const& x : inputs) {
          string output;
          for(auto i=x.second.begin(); i != x.second.end();i++) {
            output = "";
            utils.generateOutputPath(x.first,align_folder_out,output);
            string input = x.first + '/' + *i;
            output = output + '/' + *i;
            outputAlignement.push_back(output);
            #ifdef EBUG
              cout << "Input ----- " << input << "-------- Output ->" << output <<endl;
            #endif
            FaceExtracted faceModule = FaceExtracted(input,output);
            faceModule.detectFaces();
            faceModule.generateLandmark();
            faceModule.getRotatedFaces();
            #ifdef ISPLAY
            faceModule.displayResult(2);
            #endif
            faceModule.generateThumbnails(96);
          }
        }
      }
    }
    /* Step 2 : case 1: Training the classifier */
    if(train) {
      #ifdef VERBOSE
      cout << "Training option selected, starting..." << endl;
      #endif
      
        /* Step2.1 : Data augmentation */
        string alignedPath = "./";
        alignedPath += align_folder_out;
        alignedPath += '/';
        boost::filesystem::path alignedFolderPath(alignedPath);
        Utils dataAugUtil;
        dataAugUtil.listSubPath(alignedFolderPath);
        std::map<string,std::vector<string>> dataAugInputs = dataAugUtil.getFileNames();

        #pragma omp parallel num_threads(NUM_THREADS)
        {
          #pragma omp single
          for(auto const& x : dataAugInputs) {
            string output;
            
            for(auto i=x.second.begin();i!=x.second.end();i++) {
              output = x.first + '/' + *i;
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

    /* Step 2: case 1: Infering the person */
    if(infer) {
      Utils inferUtil;
      boost::filesystem::path inferPath(align_folder_out);
      inferUtil.listSubPath(inferPath);

      std::map<string, std::vector<string>> inferringFolder = inferUtil.getFileNames();
      for(auto const& x : inferringFolder) {
        for(auto i=x.second.begin();i!=x.second.end();i++) {
            inferFace(x.first + '/' + *i);
          }
      }

    }

    if(parser.has("video")) {
      VideoCapture cap(video);
      if(!cap.isOpened()) {
        cerr << "Error opening video stream" << endl;
      }
      system("mkdir outputFrame");
      int cpt = 0;
      while(1) {
        Mat frame;
        cap >> frame;
        if(frame.empty()) break;
        if(cpt < 100) {
          cpt++;
          imshow("Video",frame);
          waitKey(1);
          continue;
        } else cpt=0;
        
       
        
        /* Align faces in frame */
        
        FaceExtracted faceModule = FaceExtracted(frame,"./outputFrame/smith.jpg");
        faceModule.detectFaces();
        faceModule.generateLandmark();
        faceModule.getRotatedFaces();
        #ifdef ISPLAY
          faceModule.displayResult(2);
        #endif
        faceModule.generateThumbnails(96);
        
        Utils inferUtil;
        boost::filesystem::path inferPath("./outputFrame");
        inferUtil.listSubPath(inferPath);

        std::map<string, std::vector<string>> inferringFolder = inferUtil.getFileNames();
        for(auto const& x : inferringFolder) {
          for(auto i=x.second.begin();i!=x.second.end();i++) {
              inferFace(x.first + '/' + *i);
          }
        }
       
      }
      system("rm ./outputFrame/*");
      cap.release();
      destroyAllWindows();
     }

    return 0;
}
