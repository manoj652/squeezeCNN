//
// Created by Naif on 04/06/2018.
//

#ifndef SQUEEZECNN_UTILS_H
#define SQUEEZECNN_UTILS_H

#pragma once

#include <time.h>
#include <string>
#if (_MSC_VER != 1600)
#include <chrono>
#else
#include <time.h>
#endif
#include "core_math.h"
#include "network.h"

#ifdef SQUEEZECNN_CV2
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"

#pragma comment(lib, "opencv_core249")
#pragma comment(lib, "opencv_highgui249")
#pragma comment(lib, "opencv_imgproc249")
#pragma comment(lib, "opencv_contrib249")
#endif
#ifdef SQUEEZECNN_CV3
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#pragma comment(lib, "opencv_world310")
#endif

namespace squeezeCNN {

    class Progress {
    public:
        Progress(int size=-1,const char *label = NULL) {
            reset(size,label);
        }
    };


} //end namespace

#endif //SQUEEZECNN_UTILS_H
