#include "face_extraction.hpp"


using namespace std;
using namespace cv;
using namespace cv::face;

#define COLOR_LANDMARK Scalar(255,100,100)

FaceExtracted::FaceExtracted(Mat frame)  {
  motherFrame = frame;
  face_cascade_name = "/home/naif/Documents/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
  window_name = "Face extractor";

}

void FaceExtracted::detectFaces() {
  Mat frame_gray;
  cvtColor(motherFrame,frame_gray, COLOR_BGR2GRAY);
  equalizeHist(frame_gray,frame_gray);

  if(!_faceCascade.load(face_cascade_name)) return;

  _faceCascade.detectMultiScale(frame_gray,_faces, 1.1,2,0|CASCADE_SCALE_IMAGE, Size(60,60));
  Mat faceROI;

  for(size_t i=0;i<_faces.size(); i++) {
    _faces[i].width = _faces[i].width + PADDING;
    _faces[i].height = _faces[i].height + PADDING;
    faceROI = motherFrame( _faces[i] );
    facesROI.push_back(faceROI);
  }
}

void FaceExtracted::alignDetectedFace(int index) {
   float leftEyeX, leftEyeY;
   float rightEyeX, rightEyeY;

   /* Landmarks index 36 to 41 for left eye and 42 to 47 for right */
   Point leftEyeStart(_landmarks[index][36].x,_landmarks[index][36].y);
   Point leftEyeEnd(_landmarks[index][41].x,_landmarks[index][41].y);
   Point rightEyeStart(_landmarks[index][42].x,_landmarks[index][42].y);
   Point leftEyeEnd(_landmarks[index][47].x,_landmarks[index][47].y);

   /* Finding center of left eye */
   float X,Y,averageX,averageY;
   int cpt;
   for(uint8_t i=36; i<=41;i++) {
     X += _landmarks[index][i].x;
     Y += _landmarks[index][i].y;
     cpt++;
   }
   averageX = X/cpt;
   averageY = Y/cpt;
   Point centerLeft(averageX,averageY);
   for(uint8_t i=42; i<=47;i++) {
     X += _landmarks[index][i].x;
     Y += _landmarks[index][i].y;
     cpt++;
   }
   averageX = X/cpt;
   averageY = Y/cpt;
   Point centerRight(averageX,averageY);

   int dY = centerRight.y - centerLeft.y;
   int dX = centerRight.x - centerRight.y;

}

void FaceExtracted::saveCroppedFaces(string path) {
  for(size_t i=0;i<facesROI.size();i++) {
    stringstream ssfn;
    ssfn << path << '/' << i << ".jpg";
    string filename = ssfn.str();
    imwrite(filename,facesROI[i]);
  }
}

void FaceExtracted::drawPolyline(const int index,const int start, const int end, bool isClosed = false) {
  vector<Point> points;
  for(int i = start;i<=end;i++) {
    points.push_back(Point((_landmarks[index])[i].x,(_landmarks[index])[i].y));
  }
  polylines(motherFrame,points,isClosed,COLOR_LANDMARK,2,16);
}

int FaceExtracted::generateLandmark() {
  Ptr<Facemark> facemark = FacemarkLBF::create();
  facemark->loadModel("models/lbfmodel.yaml");

  bool success = facemark->fit(motherFrame,_faces,_landmarks);

  if(success) return 0;
  else return -1;
}

void FaceExtracted::displayResult(int action) {
  switch(action) {
    case 0:
    while(1) {
      if(waitKey(10) == 27) break;
      for(size_t i=0;i<_faces.size();i++) {
        Point top(_faces[i].x-PADDING,_faces[i].y-PADDING);
        Point bottom(_faces[i].x+_faces[i].width+PADDING,_faces[i].y+_faces[i].height+PADDING);
        rectangle(motherFrame,top,bottom,Scalar(255,255,0),3,8,0);
        imshow(window_name,motherFrame);
      }
    }

    case 2: //Drawing the _landmarks
    for(int i=0;i<_landmarks.size();i++) {
      //i loop through the faces, j through the _landmarks
      if(_landmarks[i].size() == 68) {
        drawPolyline(i,0, 16);           // Jaw line
        drawPolyline(i,17, 21);          // Left eyebrow
        drawPolyline(i,22, 26);          // Right eyebrow
        drawPolyline(i,27, 30);          // Nose bridge
        drawPolyline(i,30, 35, true);    // Lower nose
        drawPolyline(i,36, 41, true);    // Left eye
        drawPolyline(i,42, 47, true);    // Right Eye
        drawPolyline(i,48, 59, true);    // Outer lip
        drawPolyline(i,60, 67, true); // Inner lip
      } else {
        for(int j = 0; j < _landmarks[i].size(); j++) {
           circle(motherFrame,_landmarks[i][j],3, COLOR_LANDMARK, FILLED);
         }
      }

    }
    while(1) {
      if(waitKey(10) == 27) break;
      imshow(window_name,motherFrame);
    }
    default:
    return;
  }
}
