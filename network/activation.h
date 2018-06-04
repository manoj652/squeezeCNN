//
// Created by Naif on 04/06/2018.
//

#ifndef SQUEEZECNN_ACTIVATION_H
#define SQUEEZECNN_ACTIVATION_H

#pragma once

#include <math.h>
#include <algorithm>
#include <string>

namespace squeezeCNN {

    namespace identity {
        inline void f(float *in, const int size, const float *bias) {
            for(int i=0;i<size;i++) {
                in[i] = (in[i] + bias[i]);
            }
        }

        inline void fc(float *in, const int size, const float bias) {
            for(int i=0;i<size;i++){
                in[i] = (in[i] + bias);
            }
        }

        inline float df(float *in, const int size, const float *bias) {
            return 1.f;
        }
        const char name[] = "identity";
    }

    namespace relu {
        inline void f(float *in, const int size, const float *bias) {
            for(int i=0; i<size; i++) {
                if((in[i] + bias[i]) < 0) in[i] = 0;
                else in[i] = in[i] + bias[i];
            }
        }

        inline void fc(float *in, const int size, const float bias) {
            for(int i=0;i<size;i++) {
                if((in[i] + bias) < 0) in[i] = 0;
                else in[i] = in[i] + bias;
            }
        }

        inline float df(float *in, int i, const int size) {
            if(in[i] > 0) return 1.0f;
            else return 0.0f;
        }

        const char name[] = "relu";
    }

    namespace lrelu {
        inline void f(float *in, const int size, const float *bias) {
            for(int i=0; i<size; i++) {
                if((in[i] + bias[i]) < 0) in[i] = 0.01f*(in[i] + bias[i]);
                else in[i] = in[i] + bias[i];
            }
        }

        inline void fc(float *in, const int size, const float bias) {
            for(int i=0; i<size; i++) {
                if((in[i] + bias) < 0) in[i] = 0.01f*(in[i] + bias);
                else in[i] = in[i] + bias;
            }
        }

        inline float df(float *in, int i, const int size) {
            if(in[i] > 0)
                return 1.0f;
            else return 0.01f;
        }
        const char name[] = "lrelu";
    }

    namespace sigmoid {
        //TODO: To be optimized with a LUT
        inline void f(float *in, const int size, const float *bias) {
            float max = in[0];
            for(int j=1; j<size; j++) {
                if(in[j]> max) max = in[j];
            }

            float deno = 0;
            for(int j=0;j<size;j++) {
                deno += std::exp(in[j] - max);
            }

            for(int i=0;i<size;i++) in[i] = std::exp(in[i] - max) / deno;
        }

        inline void fc(float *in, const int size, const float bias) {
            float max = in[0];
            for(int j=1; j<size; j++) {
                if(in[j]> max) max = in[j];
            }

            float deno = 0;
            for(int j=0;j<size;j++) {
                deno += std::exp(in[j] - max);
            }

            for(int i=0;i<size;i++) in[i] = std::exp(in[i] - max) / deno;
        }
        inline float df(float *in, int i const int size) {
            //TODO: stopped here
        }

    }

    namespace fastSigmoid {

        inline void f(float *in, const int size, const float *bias) {

        }

    }
}
#endif //SQUEEZECNN_ACTIVATION_H
