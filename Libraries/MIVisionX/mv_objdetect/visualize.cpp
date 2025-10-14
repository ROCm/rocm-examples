// MIT License
//
// Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "visualize.h"

using namespace cv;
using namespace std;

Visualize::Visualize(float confidence) : m_confidence(confidence) {}

Visualize::~Visualize() {}

void Visualize::show(const Mat& img, std::vector<BBox>& results, int batch_size)
{
    // Create Mat from data
    int   detected_num = (int)results.size();
    int   img_height   = img.rows;
    int   img_width    = img.cols;
    float x_offs = 0.f, y_offs = 0.f;

    if(batch_size == 4)
    {
        img_height >>= 1;
        img_width >>= 1;
        x_offs = (float)img_width;
        y_offs = (float)img_height;
        for(int i = 0; i < detected_num; i++)
        {
            BBox* pb = &results[i];
            if(pb->confidence > m_confidence)
            {
                float w2     = pb->w / 2.f;
                float h2     = pb->h / 2.f;
                float left   = (pb->x - w2) * img_width;
                float right  = (pb->x + w2) * img_width;
                float top    = (pb->y - h2) * img_height;
                float bottom = (pb->y + h2) * img_height;
                if(left < 0.0)
                {
                    left = 0.0;
                }
                if(right > img.cols - 1)
                {
                    right = (float)(img_width - 1);
                }
                if(top < 0.0)
                {
                    top = 0;
                }
                if(bottom > img_height - 1)
                {
                    bottom = (float)(img_height - 1);
                }
                if(pb->imgnum == 1)
                {
                    left += x_offs;
                    right += x_offs;
                }
                else if(pb->imgnum == 2)
                {
                    top += y_offs;
                    bottom += y_offs;
                }
                else if(pb->imgnum == 3)
                {
                    left += x_offs;
                    right += x_offs;
                    top += y_offs;
                    bottom += y_offs;
                }
                int    index = pb->label;
                Scalar clr(colors[index][0], colors[index][1], colors[index][2]);
                string txt = yoloClasses[index];
                rectangle(img, Point((int)left, (int)top), Point((int)right, (int)bottom), clr, 2);
                Size size   = getTextSize(txt, FONT_HERSHEY_COMPLEX_SMALL, 0.8, 2, 0);
                int  width  = size.width;
                int  height = size.height;
                rectangle(img,
                          Point((int)left, ((int)top - (height + 4))),
                          Point(((int)left + width), (int)top),
                          clr,
                          -1);
                putText(img,
                        txt,
                        Point((int)left, (int)top),
                        FONT_HERSHEY_COMPLEX_SMALL,
                        0.8,
                        Scalar(255, 255, 255),
                        1,
                        8);
            }
        }
    }
    else if(batch_size > 4)
    {
        img_height >>= (batch_size == 8) ? 1 : 2;
        img_width >>= 2;
        x_offs = (float)img_width;
        y_offs = (float)img_height;
        for(int i = 0; i < detected_num; i++)
        {
            BBox* pb = &results[i];
            if(pb->confidence > m_confidence)
            {
                float w2     = pb->w / 2.f;
                float h2     = pb->h / 2.f;
                float left   = (pb->x - w2) * img_width;
                float right  = (pb->x + w2) * img_width;
                float top    = (pb->y - h2) * img_height;
                float bottom = (pb->y + h2) * img_height;
                if(left < 0.0)
                {
                    left = 0.0;
                }
                if(right > img.cols - 1)
                {
                    right = (float)(img_width - 1);
                }
                if(top < 0.0)
                {
                    top = 0;
                }
                if(bottom > img_height - 1)
                {
                    bottom = (float)(img_height - 1);
                }
                x_offs = (float)img_width * (pb->imgnum & 0x3);
                y_offs = (float)img_height * ((pb->imgnum & 0xc) >> 2);
                left += x_offs;
                right += x_offs;
                top += y_offs;
                bottom += y_offs;
                int    index = pb->label;
                Scalar clr(colors[index][0], colors[index][1], colors[index][2]);
                string txt = yoloClasses[index];
                rectangle(img, Point((int)left, (int)top), Point((int)right, (int)bottom), clr, 2);
                Size size   = getTextSize(txt, FONT_HERSHEY_COMPLEX_SMALL, 0.8, 2, 0);
                int  width  = size.width;
                int  height = size.height;
                rectangle(img,
                          Point((int)left, ((int)top - (height + 4))),
                          Point(((int)left + width), (int)top),
                          clr,
                          -1);
                putText(img,
                        txt,
                        Point((int)left, (int)top),
                        FONT_HERSHEY_COMPLEX_SMALL,
                        0.8,
                        Scalar(255, 255, 255),
                        1,
                        8);
            }
        }
    }
    else
    {
        for(int i = 0; i < detected_num; i++)
        {
            BBox* pb = &results[i];
            if(pb->confidence > m_confidence)
            {
                float w2     = pb->w / 2.f;
                float h2     = pb->h / 2.f;
                float left   = (pb->x - w2) * img_width;
                float right  = (pb->x + w2) * img_width;
                float top    = (pb->y - h2) * img_height;
                float bottom = (pb->y + h2) * img_height;
                if(left < 0.0)
                {
                    left = 0.0;
                }
                if(right > img.cols - 1)
                {
                    right = (float)(img_width - 1);
                }
                if(top < 0.0)
                {
                    top = 0;
                }
                if(bottom > img_height - 1)
                {
                    bottom = (float)(img_height - 1);
                }
                int    index = pb->label;
                Scalar clr(colors[index][0], colors[index][1], colors[index][2]);
                string txt = yoloClasses[index];
                rectangle(img, Point((int)left, (int)top), Point((int)right, (int)bottom), clr, 2);
                Size size   = getTextSize(txt, FONT_HERSHEY_COMPLEX_SMALL, 0.8, 2, 0);
                int  width  = size.width;
                int  height = size.height;
                rectangle(img,
                          Point((int)left, ((int)top - (height + 4))),
                          Point(((int)left + width), (int)top),
                          clr,
                          -1);
                putText(img,
                        txt,
                        Point((int)left, (int)top),
                        FONT_HERSHEY_COMPLEX_SMALL,
                        0.8,
                        Scalar(255, 255, 255),
                        1,
                        8);
            }
        }
    }
    imshow("Detected Image", img);
}

void Visualize::legend_image()
{
    string window_name = "AMD Object Detection - Legend";

    Size legend_geometry = Size(325, (20 * 40) + 40);
    Mat  legend          = Mat::zeros(legend_geometry, CV_8UC3);
    Rect roi             = Rect(0, 0, 325, (20 * 40) + 40);
    legend(roi).setTo(Scalar(255, 255, 255));

    for(int l = 0; l < 20; l++)
    {
        Scalar clr(colors[l][0], colors[l][1], colors[l][2]);
        string class_name = yoloClasses[l];
        putText(legend,
                class_name,
                Point(20, (l * 40) + 30),
                cv::FONT_HERSHEY_SIMPLEX,
                0.6,
                Scalar(0, 0, 0),
                1,
                8);
        rectangle(legend, Point(225, (l * 40)), Point(300, (l * 40) + 40), clr, -1);
    }
    imshow(window_name, legend);

    return;
}
