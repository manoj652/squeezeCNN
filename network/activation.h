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
            for (int i = 0; i < size; i++) {
                in[i] = (in[i] + bias[i]);
            }
        }

        inline void fc(float *in, const int size, const float bias) {
            for (int i = 0; i < size; i++) {
                in[i] = (in[i] + bias);
            }
        }

        inline float df(float *in, const int size, const float *bias) {
            return 1.f;
        }

        const char name[] = "identity";
    };

    namespace relu {
        inline void f(float *in, const int size, const float *bias) {
            for (int i = 0; i < size; i++) {
                if ((in[i] + bias[i]) < 0) in[i] = 0;
                else in[i] = in[i] + bias[i];
            }
        }

        inline void fc(float *in, const int size, const float bias) {
            for (int i = 0; i < size; i++) {
                if ((in[i] + bias) < 0) in[i] = 0;
                else in[i] = in[i] + bias;
            }
        }

        inline float df(float *in, int i, const int size) {
            if (in[i] > 0) return 1.0f;
            else return 0.0f;
        }

        const char name[] = "relu";
    };

    namespace lrelu {
        inline void f(float *in, const int size, const float *bias) {
            for (int i = 0; i < size; i++) {
                if ((in[i] + bias[i]) < 0) in[i] = 0.01f * (in[i] + bias[i]);
                else in[i] = in[i] + bias[i];
            }
        }

        inline void fc(float *in, const int size, const float bias) {
            for (int i = 0; i < size; i++) {
                if ((in[i] + bias) < 0) in[i] = 0.01f * (in[i] + bias);
                else in[i] = in[i] + bias;
            }
        }

        inline float df(float *in, int i, const int size) {
            if (in[i] > 0)
                return 1.0f;
            else return 0.01f;
        }

        const char name[] = "lrelu";
    };

    namespace sigmoid {
        //TODO: To be optimized with a LUT
        inline void f(float *in, const int size, const float *bias) {
            float max = in[0];
            for (int j = 1; j < size; j++) {
                if (in[j] > max) max = in[j];
            }

            float deno = 0;
            for (int j = 0; j < size; j++) {
                deno += std::exp(in[j] - max);
            }

            for (int i = 0; i < size; i++) in[i] = std::exp(in[i] - max) / deno;
        }

        inline void fc(float *in, const int size, const float bias) {
            float max = in[0];
            for (int j = 1; j < size; j++) {
                if (in[j] > max) max = in[j];
            }

            float deno = 0;
            for (int j = 0; j < size; j++) {
                deno += std::exp(in[j] - max);
            }

            for (int i = 0; i < size; i++) in[i] = std::exp(in[i] - max) / deno;
        }

        inline float df(float *in, int i, const int size) {
            return in[i] * (1.f - in[i]);
        }

    };

    namespace fastSigmoid {

        inline void f(float *in, const int size, const float *bias) {
            for (int i = 0; i < size; i++) {
                in[i] = in[i] / (1 + std::abs(in[i]));
            }

        }
        //TODO: finish sigmoid

    };

    namespace softmax {
        inline void f(float *in, const int size, const float *bias) {
            float max = in[0];
            for (int j = 1; j < size; j++)
                if (in[j] > max) max = in[j];

            float denom = 0;
            for (int j = 0; j < size; j++)
                denom += std::exp(in[j] - max);

            for (int i = 0; i < size; i++)
                in[i] = std::exp(in[i] - max) / denom;
        }

        inline void fc(float *in, const int size, const float bias) {
            float max = in[0];
            for (int j = 1; j < size; j++) if (in[j] > max) max = in[j];

            float denom = 0;
            for (int j = 0; j < size; j++)
                denom += std::exp(in[j] - max);

            for (int i = 0; i < size; i++)
                in[i] = std::exp(in[i] - max) / denom;
        }

        inline float df(float *in, int i, const int size) {
            // don't really use... should use good cost func to make this go away
            return in[i] * (1.f - in[i]);

        }

        const char name[] = "softmax";
    };

    namespace none {
        inline void f(float *in, const int size, const float *bias) {
            return;
        }

        inline void fc(float *in, const int size, const float bias) {
            return;
        }

        inline float df(float *in, int i, int size) {
            return 0;
        }

        const char name[] = "none";
    };

    typedef struct {
    public:
        void (*f)(float *, const int, const float *);

        void (*fc)(float *, const int, const float);

        void (*df)(float *, int, const int);
    } activation_function;

    activation_function *new_activation_function(std::string act) {
        activation_function *p = new activation_function;
        if (act.compare(identity::name) == 0) {
            p->f = &identity::f;
            p->fc = &identity::fc;
            p->df = &identity::df;
            p->name = identity::name;
            return p;
        }
        if (act.compare(lrelu::name) == 0) {
            p->f = &lrelu::f;
            p->fc = &lrelu::fc;
            p->df = &lrelu::df;
            p->name = lrelu::name;
            return p;
        }
        if (act.compare(relu::name) == 0) {
            p->f = &relu::f;
            p->fc = &relu::fc;
            p->df = &relu::df;
            p->name = relu::name;
            return p;
        }
        if (act.compare(sigmoid::name) == 0) {
            p->f = &sigmoid::f;
            p->fc = &sigmoid::fc;
            p->df = &sigmoid::df;
            p->name = sigmoid::name;
            return p;
        }
        if (act.compare(none::name) == 0) {
            p->f = &none::f;
            p->fc = &none::fc;
            p->df = &none::df;
            p->name = none::name;
            return p;
        }
        if (act.compare(softmax::name) == 0) {
            p->f = &softmax::f;
            p->fc = &softmax::fc;
            p->df = &softmax::df;
            p->name = softmax::name;
            return p;
        }
        delete p;
        return NULL;
    }

    activation_function *new_activation_function(const char *type) {
        std::string act(type);
        return new_activation_function(act);
    }
}
#endif //SQUEEZECNN_ACTIVATION_H
