//
// Created by Naif on 04/06/2018.
//

#ifndef SQUEEZECNN_CORE_MATH_H
#define SQUEEZECNN_CORE_MATH_H

#pragma once

#include <math.h>
#include <string.h>
#include <string>
#include <cstdlib>
#include <random>
#include <algorithm>


namespace squeezeCNN {

    enum padding_type {zero=0, edge =1,median_edge = 2};

    /*
     * dot function to ouput dot product bertween vectors
     * inline allows optimization by integrating code at compil time
     */
    inline float dot(const float *x1, const float *x2, const int size) {
        switch(size) {
            case 1:
                return x1[0] * x2[0];
            case 2:
                return x1[0] * x2[0] + x1[1] * x2[1];
            case 3:
                return x1[0] * x2[0] + x1[1] * x2[1] + x1[2] * x2[2];
            case 4:
                return x1[0] * x2[0] + x1[1] * x2[1] + x1[2] * x2[2] + x1[3] * x2[3];
            case 5:
                return x1[0] * x2[0] + x1[1] * x2[1] + x1[2] * x2[2] + x1[3] * x2[3] + x1[4] * x2[4];
            default:
                float tmp = 0;
                for(int i=0;i<size;i++) {
                    tmp += x1[i]*x2[i];
                }
                return tmp;
        };
    }

    /*
     * Computing dot for 2d Matrix using previous vector dot product
     */
    inline float unwrap_dot_2d(const float *x1, const float *x2, const int size, int stride1,
                                int stride2) {
        float tmp = 0;
        for(int i=0;i<size;i++)
            tmp += dot(&x1[stride1*i],&x2[stride2*i],size);
        return tmp;
    }

    /*
     * Convolution requires 180 degree rotation for weight kernel manipulation
     * Switch optimzing, as sums are less ressource heavy than loops that adds to complexity
     */
    inline float dot_180rot(const float *x1,const float *x2, const int size) {
        switch(size) {
            case 1:
                return x1[0]*x2[0];
            case 2:
                return x1[0]*x2[1]+x1[1]*x2[0];
            case 3:
                return x1[0]*x2[2]+x1[1]*x2[1]+x1[2]*x2[0];
            case 4:
                return x1[0]*x2[3]+x1[1]*x2[2]+x1[2]*x2[1]+x1[3]*x2[0];
            case 5:
                return x1[0]*x2[4]+x1[1]*x2[3]+x1[2]*x2[2]+x1[3]*x2[1]+x1[4]*x2[0];

            default:
                float tmp=0;
                for(int i=0;i<size;i++) {
                    tmp += x1[i] * x2[size-i-1];
                }
                return tmp;
        };
    }

    /*
     * Matrix dot product using a 180 degree rotation
     */
    inline float unwrap_2d_dot_rot180(const float *x1, const float *x2, const int size, int stride1, int stride2) {
        float tmp=0;
        for(int i=0;i<size;i++) {
            tmp += dot_180rot(&x1[stride1*i],&x2[stride2*i],size); /* Performing 180 degree multiplication */
        }

        return tmp;
    }

    /*
     * Util function to unwrap a NxN Matrix
     */
    inline void unwrap_aligned_Matrix(const int N, float *aligned_out,const float *in, const int in_size, const int stride=1) {
        const int node_size = (in_size - N)+1;
        int c1 = 0;
        int off = 0;
        const int inc_off = N*N*8;
        for(int i=0;i<node_size;i++) {
            for(int j=0;j<node_size;j++) {
                const float *tp = in + j*in_size + i;
                if(N==5) {
                    for(int k=0;k<5;k++) {
                        aligned_out[c1 + 0 +  k * 40 + off] = tp[0 + 0 + in_size*k];
                        aligned_out[c1 + 8 +  k * 40 + off] = tp[0 + 1 + in_size*k];
                        aligned_out[c1 + 16 + k * 40 + off] = tp[0 + 2 + in_size*k];
                        aligned_out[c1 + 24 + k * 40 + off] = tp[0 + 3 + in_size*k];
                        aligned_out[c1 + 32 + k * 40 + off] = tp[0 + 4 + in_size*k];
                    }
                } else if(N==3) {
                    aligned_out[c1 + off] = tp[0];
                    aligned_out[c1 + 8 + off] = tp[0 + 1];
                    aligned_out[c1 + 16 + off] = tp[0 + 2];

                    aligned_out[c1 + 24 + off] = tp[0 +     in_size];
                    aligned_out[c1 + 32 + off] = tp[0 + 1 + in_size];
                    aligned_out[c1 + 40 + off] = tp[0 + 2 + in_size];

                    aligned_out[c1 + 48 + off] = tp[0 +     2 * in_size];
                    aligned_out[c1 + 56 + off] = tp[0 + 1 + 2 * in_size];
                    aligned_out[c1 + 64 + off] = tp[0 + 2 + 2 * in_size];
                } else {
                    int cnt=0;
                    for(int k=0;k<N;k++) {
                        for(int m=0;m<N;m++) {
                            aligned_out[c1 + cnt*8 + off] = tp[m + in_size*k]; //Writing 32 bits at a time to optimize write buff
                            cnt++;
                        }
                    }
                }
                off++;
                if(off>7) {
                    off = 0;
                    c1 += inc_off;
                }
            }
        }
    }

    /*
     * Dot sum on unwrapped NxN Matrix used for optimization. Reducing loop complexity from o(N*N) to o(2N)
     */
    inline void dotsum_unwrapped_NxN(const int N,const float *_img, const float *filter_ptr, float *out, const int outsize) {
        const int dim = N*N;
        for(int i=0;i<outsize;i+=8) {
            float *c = out+i;
            for(int j=0;j<dim;j++) {
                const float f = filter_ptr[j];
                c[0] += _img[0] *f;
                c[1] += _img[1] *f;
                c[2] += _img[2] *f;
                c[3] += _img[3] *f;
                c[4] += _img[4] *f;
                c[5] += _img[5] *f;
                c[6] += _img[6] *f;
                c[7] += _img[7] *f;
                _img += 8; //Avancée de 8 cases memoires float de 4 octets
            }
        }
    }

#ifndef MOJO_AVX
    inline void dotsum_unwrapped_2x2(const float *_img, const float *filter_ptr, float *out, const int outsize) {
        dotsum_unwrapped_NxN(2,_img,filter_ptr,out,outsize);
    }
    inline void dotsum_unwrapped_3x3(const float *_img, const float *filter_ptr, float *out, const int outsize) {
        dotsum_unwrapped_NxN(3,_img,filter_ptr,out,outsize);
    }
    inline void dotsum_unwrapped4x4(const float *_img, const float *filter_ptr, float *out, const int outsize) {
        dotsum_unwrapped_NxN(4,_img,filter_ptr,out,outsize);
    }
    inline void dotsum_unwrapped_5x5(const float *_img, const float *filter_ptr, float *out, const int outsize) {
        dotsum_unwrapped_NxN(5,_img,filter_ptr,out,outsize);
    }
    inline void dotsum_unwrapped_6x6(const float *_img, const float *filter_ptr, float *out, const int outsize) {
        dotsum_unwrapped_NxN(6,_img,filter_ptr,out,outsize);
    }
    inline void dotsum_unwrapped_7x7(const float *_img, const float *filter_ptr, float *out, const int outsize) {
        dotsum_unwrapped_NxN(7,_img,filter_ptr,out,outsize);
    }
#endif

/* ----------------------------- DEFINING THE Matrix CLASS -------------------------- */

    class Matrix {
        int _size;
        int _capacity;
        float *_x_mem;

        void delete_x() {
            delete[] _x_mem;
            x= NULL;
            _x_mem = NULL;
        }

        float *new_x(const int size) {
            _x_mem = new float[size + 8 + 7];
            x = (float *)(((uintptr_t)_x_mem +32) & ~(uintptr_t)0x1F);
            return x;
        }

    public:
        std::string _name;
        int cols, rows, chans;
        int chan_stride;
        int chan_aligned;
        float *x;

        virtual int compute_chan_stride(int w, int h) {
            if(chan_aligned) {
                int s= w*h;
                const int remainer = s%8;
                if(remainer > 0) s += 8-remainer;
                return s;
            } else return h*w;
        }

        Matrix():
                cols(0),
                rows(0),
                chans(0),
                _size(0),
                _capacity(0),
                chan_stride(0),
                x(NULL),
                chan_aligned(0) {}

        Matrix(int _w,int _h, int _c=1, const float *data = NULL, int align_chan=0): cols(_w),
                rows(_w), chans(_c) {
            chan_aligned = align_chan;
            chan_stride = compute_chan_stride(cols,rows);
            _size = chan_stride*chans;
            _capacity = _size;
            x = new_x(_size);
            if(data != NULL) {
                memcpy(x,data,_size*sizeof(float));
            }
        }

        Matrix(const Matrix &m):
            cols(m.cols),
            rows(m.rows),
            chan_aligned(m.chan_aligned),
            chans(m.chans),
            chan_stride(m.chan_stride),
            _size(m._size),
            _capacity(m._size) {
            x = new_x(_size);
            memcpy(x,m.x,sizeof(float)*_size);
            }

         Matrix(const Matrix &m,int pad_cols, int pad_rows, squeezeCNN::padding_type padding =  squeezeCNN::zero, int threads=1):
                 cols(m.cols),
                 rows(m.rows),
                 chans(m.chans),
                 chan_aligned(m.chan_aligned),
                 chan_stride(m.chan_stride),
                 _size(m._size),
                 _capacity(m._size)
         {
                x= new_x(_size);
                memcpy(x,m.x,sizeof(float) * _size);
                *this = pad(pad_cols,pad_rows,padding,threads);
         }

         ~Matrix() { if(x) delete_x();}

         /* Matrix specific functions */

        Matrix get_chans(int start_channel, int num_channels=1) const {
            return Matrix(cols,rows,num_channels,&x[start_channel*chan_stride]);
        }

        Matrix pad(int dx, int dy, squeezeCNN::padding_type edge_pad = squeezeCNN::zero,int threads = 1) {
            return pad(dx,dy,dx,dy,edge_pad,threads);
        }

        Matrix pad(int dx, int dy, int dx_right, int dy_bottom,
                   squeezeCNN::padding_type edge_pad = squeezeCNN::zero, int threads = 1) const {
            Matrix v(cols+dx+dx_right,rows+dy+dy_bottom,chans);
            v.fill(0);

#pragma omp parallel for num_threads(threads) /* Used for openMP multithreading */

            for(int k=0; k<chans; k++)
            {
                const int v_chan_offset=k*v.chan_stride;
                const int chan_offset=k*chan_stride;
                // find median color of perimeter
                float median = 0.f;
                if (edge_pad == squeezeCNN::median_edge)
                {
                    int perimeter = 2 * (cols + rows - 2);
                    std::vector<float> d(perimeter);
                    for (int i = 0; i < cols; i++)
                    {
                        d[i] = x[i+ chan_offset]; d[i + cols] = x[i + cols*(rows - 1)+ chan_offset];
                    }
                    for (int i = 1; i < (rows - 1); i++)
                    {
                        d[i + cols * 2] = x[cols*i+ chan_offset];
                        d[perimeter - i] = x[cols - 1 + cols*i+ chan_offset];
                    }

                    std::nth_element(d.begin(), d.begin() + perimeter / 2, d.end());
                    median = d[perimeter / 2];
                }

                for(int j=0; j<rows; j++)
                {
                    memcpy(&v.x[dx+(j+dy)*v.cols+v_chan_offset], &x[j*cols+chan_offset], sizeof(float)*cols);
                    if(edge_pad== squeezeCNN::edge) {
                        // do left/right side
                        for(int i=0; i<dx; i++)
                            v.x[i+(j+dy)*v.cols+v_chan_offset]=x[0+j*cols+chan_offset];
                        for (int i = 0; i<dx_right; i++)
                            v.x[i + dx + cols + (j + dy)*v.cols + v_chan_offset] = x[(cols - 1) + j*cols + chan_offset];
                    }
                    else if (edge_pad == squeezeCNN::median_edge) {
                        for (int i = 0; i < dx; i++)
                            v.x[i + (j + dy)*v.cols + v_chan_offset] = median;
                        for (int i = 0; i < dx_right; i++)
                            v.x[i + dx + cols + (j + dy)*v.cols + v_chan_offset] = median;
                    }
                }
                // top bottom pad
                if(edge_pad== squeezeCNN::edge) {
                    for(int j=0; j<dy; j++)
                        memcpy(&v.x[(j)*v.cols+v_chan_offset],&v.x[(dy)*v.cols+v_chan_offset], sizeof(float)*v.cols);
                    for (int j = 0; j<dy_bottom; j++)
                        memcpy(&v.x[(j + dy + rows)*v.cols + v_chan_offset], &v.x[(rows - 1 + dy)*v.cols + v_chan_offset], sizeof(float)*v.cols);
                }
                if (edge_pad == squeezeCNN::median_edge) {
                    for (int j = 0; j<dy; j++)
                        for (int i = 0; i<v.cols; i++)
                            v.x[i + j*v.cols + v_chan_offset] = median;
                    for (int j = 0; j<dy_bottom; j++)
                        for (int i = 0; i<v.cols; i++)
                            v.x[i + (j + dy + rows)*v.cols + v_chan_offset] = median;
                }
            }

            return v;
        } //end pad method

         Matrix crop(int dx, int dy, int w, int h, int threads=1) const {
            Matrix v(w,h,chans);
#pragma omp parallel for num_threads(threads)
            for(int k=0;k<chans;k++) {
                for(int j=0;j<h;j++) {
                    memcpy(&v.x[j*w+k*v.chan_stride],&x[dx+(j+dy)*cols+k*chan_stride], sizeof(float)*w);
                }
            }

            return v;
        } // end crop method

        squeezeCNN::Matrix shift(int dx,int dy, squeezeCNN::padding_type edge_pad=squeezeCNN::zero) {
            int orig_cols = cols;
            int orig_rows = rows;
            int off_x = abs(dx);
            int off_y = abs(dy);

            squeezeCNN::Matrix shiftedMat = pad(off_x, off_y, edge_pad);

            return shiftedMat.crop(off_x-dx,off_y-dy,orig_cols,orig_rows); //shiff is simply cropping a padded Matrix
        } //emnd shift method

        squeezeCNN::Matrix flip_cols() {
            squeezeCNN::Matrix v(cols,rows,chans);
            for(int k=0;k<chans;k++) {
                for(int j=0;j<rows;j++) {
                    for(int i=0;i<cols;i++) {
                        v.x[i+j*cols+k*chan_stride]=x[(cols-i-1)+j*cols+k*chan_stride];
                    }
                }
            }

            return v;
        }

        squeezeCNN::Matrix flip_rows()
        {
            squeezeCNN::Matrix v(cols, rows, chans);

            for (int k = 0; k<chans; k++)
                for (int j = 0; j<rows; j++)
                    memcpy(&v.x[(rows-1-j)*cols + k*chan_stride],&x[j*cols + k*chan_stride], cols*sizeof(float));

            return v;
        }

        void clip(float min, float max)
        {
            int s = chan_stride*chans;
            for (int i = 0; i < s; i++)
            {
                if (x[i] < min) x[i] = min;
                if (x[i] > max) x[i]=max;
            }
        }


        void min_max(float *min, float *max, int *min_i=NULL, int *max_i=NULL)
        {
            int s = rows*cols;
            int mini = 0;
            int maxi = 0;
            for (int c = 0; c < chans; c++)
            {
                const int t = chan_stride*c;
                for (int i = t; i < t+s; i++)
                {
                    if (x[i] < x[mini]) mini = i;
                    if (x[i] > x[maxi]) maxi = i;
                }
            }
            *min = x[mini];
            *max = x[maxi];
            if (min_i) *min_i = mini;
            if (max_i) *max_i = maxi;
        }

        float mean()
        {
            const int s = rows*cols;
            int cnt = 0;// channel*s;
            float average = 0;
            for (int c = 0; c < chans; c++)
            {
                const int t = chan_stride*c;
                for (int i = 0; i < s; i++)
                    average += x[i + t];
            }
            average = average / (float)(s*chans);
            return average;
        }
        float remove_mean(int channel)
        {
            int s = rows*cols;
            int offset = channel*chan_stride;
            float average=0;
            for(int i=0; i<s; i++) average+=x[i+offset];
            average= average/(float)s;
            for(int i=0; i<s; i++) x[i+offset]-=average;
            return average;
        }

        float remove_mean()
        {
            float m=mean();
            int s = chan_stride*chans;
            //int offset = channel*s;
            for(int i=0; i<s; i++) x[i]-=m;
            return m;
        }
        void fill(float val) { for(int i=0; i<_size; i++) x[i]=val;
        }
        void fill_random_uniform(float range)
        {
            std::mt19937 gen(0);
            std::uniform_real_distribution<float> dst(-range, range);
            for (int i = 0; i<_size; i++) x[i] = dst(gen);
        }
        void fill_random_normal(float std)
        {
            std::mt19937 gen(0);
            std::normal_distribution<float> dst(0, std);
            for (int i = 0; i<_size; i++) x[i] = dst(gen);
        }


        // deep copy
        inline Matrix& operator =(const Matrix &m)
        {
            resize(m.cols, m.rows, m.chans, m.chan_aligned);
            memcpy(x,m.x,sizeof(float)*_size);
            return *this;
        }

        int  size() const {return _size;}

        void resize(int _w, int _h, int _c, int align_chans=0) {
            chan_aligned = align_chans;
            int new_stride = compute_chan_stride(_w,_h);
            int s = new_stride*_c;
            if(s>_capacity)
            {
                if(_capacity>0) delete_x(); _size = s; _capacity=_size; x = new_x(_size);
            }
            cols = _w; rows = _h; chans = _c; _size = s; chan_stride = new_stride;
        }

        // dot vector to 2d mat
        inline Matrix dot_1dx2d(const Matrix &m_2d) const
        {
            squeezeCNN::Matrix v(m_2d.rows, 1, 1);
            for(int j=0; j<m_2d.rows; j++)	v.x[j]=dot(x,&m_2d.x[j*m_2d.cols],_size);
            return v;
        }

        // +=
        inline Matrix& operator+=(const Matrix &m2){
            for(int i = 0; i < _size; i++) x[i] += m2.x[i];
            return *this;
        }
        // -=
        inline Matrix& operator-=(const Matrix &m2) {
            for (int i = 0; i < _size; i++) x[i] -= m2.x[i];
            return *this;
        }
#ifndef MOJO_AVX
        // *= float
        inline Matrix operator *=(const float v) {
            for (int i = 0; i < _size; i++) x[i] = x[i] * v;
            return *this;
        }

#endif
        // *= Matrix
        inline Matrix operator *=(const Matrix &v) {
            for (int i = 0; i < _size; i++) x[i] = x[i] * v.x[i];
            return *this;
        }
        inline Matrix operator *(const Matrix &v) {
            Matrix T(cols, rows, chans);
            for (int i = 0; i < _size; i++) T.x[i] = x[i] * v.x[i];
            return T;
        }
        // * float
        inline Matrix operator *(const float v) {
            Matrix T(cols, rows, chans);
            for (int i = 0; i < _size; i++) T.x[i] = x[i] * v;
            return T;
        }

        // + float
        inline Matrix operator +(const float v) {
            Matrix T(cols, rows, chans);
            for (int i = 0; i < _size; i++) T.x[i] = x[i] + v;
            return T;
        }

        // +
        inline Matrix operator +(Matrix m2)
        {
            Matrix T(cols,rows,chans);
            for(int i = 0; i < _size; i++) T.x[i] = x[i] + m2.x[i];
            return T;
        }

    };

} //end of namespace

#endif //SQUEEZECNN_CORE_MATH_H
