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

#include "mvdeploy.h"

#include "mv_extras_postproc.h"
#include "visualize.h"
#include "vx_amd_media.h"

#include "CmdParser/cmdparser.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#define USE_OPENCL_FOR_DECODER_OUTPUT 0

// Note: ERROR_CHECK_STATUS and ERROR_CHECK_OBJECT are defined in mvdeploy.h

// Hard coded biases for yolo_v2 and yolo_v3 (x,y) for 5 bounding boxes
const float bb_biases[10]
    = {1.08f, 1.19f, 3.42f, 4.41f, 6.63f, 11.38f, 9.42f, 5.11f, 16.62f, 10.52f};

inline int64_t clock_counter()
{
    return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

inline int64_t clock_frequency()
{
    return std::chrono::high_resolution_clock::period::den
           / std::chrono::high_resolution_clock::period::num;
}

// Implementation of preprocess_addnodes_callback_fn
static vx_status MIVID_CALLBACK preprocess_addnodes_callback_fn(
    mivid_session inf_session, vx_tensor outp_tensor, mv_preprocess_callback_args* preproc_args)
{
    if(inf_session && preproc_args)
    {
        mivid_handle hdl     = (mivid_handle)inf_session;
        vx_context   context = hdl->context;
        vx_graph     graph   = hdl->graph;
        ERROR_CHECK_OBJECT(context);
        ERROR_CHECK_OBJECT(graph);
        ERROR_CHECK_STATUS(vxLoadKernels(context, "vx_amd_media"));

        // Query outp_tensor for dims
        vx_size num_dims;
        vx_size tens_dims[4] = {1, 1, 1, 1};
        ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)outp_tensor,
                                         VX_TENSOR_NUMBER_OF_DIMS,
                                         &num_dims,
                                         sizeof(num_dims)));
        if(num_dims != 4)
        {
            printf("preprocess_addnodes_callback_fn:: outp_tensor num_dims=%ld (must be 4)\n",
                   num_dims);
            return VX_ERROR_INVALID_DIMENSION;
        }
        ERROR_CHECK_STATUS(
            vxQueryTensor(outp_tensor, VX_TENSOR_DIMS, tens_dims, sizeof(tens_dims)));

#if !USE_OPENCL_FOR_DECODER_OUTPUT
        vx_image dec_image
            = vxCreateImage(context, tens_dims[0], tens_dims[1] * tens_dims[3], VX_DF_IMAGE_RGB);
        vx_node node_decoder = amdMediaDecoderNode(graph,
                                                   preproc_args->inp_string_decoder,
                                                   dec_image,
                                                   (vx_array) nullptr,
                                                   preproc_args->loop_decode);
        ERROR_CHECK_OBJECT(node_decoder);
#else
        vx_imagepatch_addressing_t addr_in = {0};
        addr_in.dim_x                      = tens_dims[0];
        addr_in.dim_y                      = tens_dims[1];
        addr_in.stride_x                   = tens_dims[3];
        addr_in.stride_y                   = tens_dims[0] * tens_dims[3];
        if(addr_in.stride_y == 0)
        {
            addr_in.stride_y = addr_in.stride_x * addr_in.dim_x;
        }
        vx_image dec_image
            = vxCreateVirtualImage(graph, addr_in.dim_x, addr_in.dim_y, VX_DF_IMAGE_RGB);
        ERROR_CHECK_OBJECT(dec_image);
        vx_node node_decoder = amdMediaDecoderNode(graph,
                                                   preproc_args->inp_string_decoder,
                                                   dec_image,
                                                   (vx_array) nullptr,
                                                   preproc_args->loop_decode,
                                                   true);
        ERROR_CHECK_OBJECT(node_decoder);
#endif
        vx_node node_img_tensor = vxConvertImageToTensorNode(graph,
                                                             dec_image,
                                                             outp_tensor,
                                                             preproc_args->preproc_a,
                                                             preproc_args->preproc_b,
                                                             0);
        ERROR_CHECK_OBJECT(node_img_tensor);
        ERROR_CHECK_STATUS(vxReleaseNode(&node_decoder));
        ERROR_CHECK_STATUS(vxReleaseNode(&node_img_tensor));
        hdl->inp_image = dec_image;
        return VX_SUCCESS;
    }
    else
    {
        printf("preprocess_addnodes_callback_fn:: inf_session not valid\n");
        return VX_FAILURE;
    }
}

int main(int argc, char** argv)
{
    // Parse command line arguments
    cli::Parser parser(argc, argv);
    parser.set_optional<std::string>("input",
                                     "input",
                                     std::string(EXAMPLE_DATA_DIR) + "/images/img_04.JPG",
                                     "Input image/video file (.jpg, .png, .mp4, .m4v)");
    parser.set_optional<std::string>("output", "output", "-", "Output file (- for no output)");
    parser.set_optional<int>("classes", "classes", 20, "Number of object classes");
    parser.set_optional<float>("conf_threshold",
                               "conf-threshold",
                               0.2f,
                               "Confidence threshold for detection");
    parser.set_optional<float>("nms_threshold",
                               "nms-threshold",
                               0.4f,
                               "NMS threshold for detection");
    parser.set_optional<int>("frames", "frames", 0, "Number of frames to process (0=all, -1=loop)");
    parser.set_optional<int>("iterations",
                             "iterations",
                             1,
                             "Number of iterations for performance testing");
    parser.set_optional<bool>("hwdec", "hwdec", false, "Use hardware decoding");
    parser.set_optional<bool>("visualize", "visualize", false, "Visualize detection results");
    parser.set_optional<int>("argmax_topk",
                             "argmax-topk",
                             0,
                             "Output argmax top-K results (0=disabled)");
    parser.set_optional<std::string>("labels", "labels", "", "Path to labels file");
    parser.run_and_exit_if_error();

    // Get parsed arguments
    const std::string input_file_name  = parser.get<std::string>("input");
    const std::string output_file_name = parser.get<std::string>("output");
    const int         num_classes      = parser.get<int>("classes");
    const float       conf_th          = parser.get<float>("conf_threshold");
    const float       nms_th           = parser.get<float>("nms_threshold");
    const int         frames_arg       = parser.get<int>("frames");
    const int         num_iterations   = parser.get<int>("iterations");
    const bool        use_vaapi        = parser.get<bool>("hwdec");
    const bool        visualize        = parser.get<bool>("visualize");
    const int         argmax_topk      = parser.get<int>("argmax_topk");
    const std::string labels_file      = parser.get<std::string>("labels");

    // Process frames argument
    bool capture_till_eof = (frames_arg == 0);
    bool loop_decode      = (frames_arg == -1);
    int  end_frame        = (frames_arg > 0) ? frames_arg : 1;
    if(loop_decode)
    {
        capture_till_eof = true;
    }

    // Load labels if provided
    std::string label_text[1000];
    if(!labels_file.empty())
    {
        std::ifstream label_stream(labels_file);
        if(!label_stream.is_open())
        {
            std::cerr << "ERROR: unable to open labels file: " << labels_file << std::endl;
            return -1;
        }
        std::string line;
        int         line_num = 0;
        while(getline(label_stream, line) && line_num < 1000)
        {
            label_text[line_num++] = line;
        }
        label_stream.close();
    }

    // Initialize deployment
    std::string install_folder = MV_INSTALL_FOLDER;
    mv_status   status;
    if((status = mvInitializeDeployment(install_folder.c_str())))
    {
        printf("ERROR: mvInitializeDeployment failed with status %d install_folder: %s\n",
               status,
               install_folder.c_str());
        return -1;
    }

    int         num_inputs  = 1;
    int         num_outputs = 1;
    const char* inout_config;
    if((status = QueryInference(&num_inputs, &num_outputs, &inout_config)))
    {
        printf("ERROR: QueryInference returned status %d\n", status);
        return -1;
    }

    float*        inp_mem = nullptr;
    float*        out_mem = nullptr;
    size_t        inp_dims[4];
    size_t        out_dims[4];
    mivid_session inf_session;
    mivid_handle  inf_hdl;
    vx_image      inp_img;
    int           do_preprocess = 0;
    float         scale_factor  = 1.0f / 255.0f;
    float         add_factor    = 0.0f;

    // Parse input and output dimensions from inout_config
    std::stringstream                           inout_dims(inout_config);
    std::vector<std::string>                    config_vec;
    std::string                                 in_names[num_inputs];
    std::string                                 out_names[num_outputs];
    std::vector<std::tuple<int, int, int, int>> input_dims;
    std::vector<std::tuple<int, int, int, int>> output_dims;
    std::string                                 substr;

    while(inout_dims.good())
    {
        getline(inout_dims, substr, ';');
        if(!substr.empty())
        {
            config_vec.push_back(substr);
        }
    }

    int in_num  = 0;
    int out_num = 0;
    int n, c, h, w;
    for(size_t i = 0; i < config_vec.size(); i++)
    {
        std::stringstream ss(config_vec[i]);
        getline(ss, substr, ',');
        if((substr.compare(0, 5, "input") == 0))
        {
            getline(ss, substr, ',');
            in_names[in_num] = substr;
            getline(ss, substr, ',');
            w = atoi(substr.c_str());
            getline(ss, substr, ',');
            h = atoi(substr.c_str());
            getline(ss, substr, ',');
            c = atoi(substr.c_str());
            getline(ss, substr, ',');
            n = atoi(substr.c_str());
            printf("Config_input::<%d %d %d %d>:%s\n", w, h, c, n, in_names[in_num].c_str());
            input_dims.push_back(std::tuple<int, int, int, int>(w, h, c, n));
            in_num++;
        }
        else if((substr.compare(0, 6, "output") == 0))
        {
            getline(ss, substr, ',');
            out_names[out_num] = substr;
            getline(ss, substr, ',');
            w = atoi(substr.c_str());
            getline(ss, substr, ',');
            h = atoi(substr.c_str());
            getline(ss, substr, ',');
            c = atoi(substr.c_str());
            getline(ss, substr, ',');
            n = atoi(substr.c_str());
            printf("Config_output::<%d %d %d %d>:%s\n", w, h, c, n, out_names[out_num].c_str());
            output_dims.push_back(std::tuple<int, int, int, int>(w, h, c, n));
            out_num++;
        }
    }

    if(input_dims.size() == 0 || output_dims.size() == 0)
    {
        printf("ERROR: Couldn't get input and output dims %d %d\n",
               (int)input_dims.size(),
               (int)output_dims.size());
        return -1;
    }

    inp_dims[3] = std::get<0>(input_dims[0]);
    inp_dims[2] = std::get<1>(input_dims[0]);
    inp_dims[1] = std::get<2>(input_dims[0]);
    inp_dims[0] = std::get<3>(input_dims[0]);
    out_dims[3] = std::get<0>(output_dims[0]);
    out_dims[2] = std::get<1>(output_dims[0]);
    out_dims[1] = std::get<2>(output_dims[0]);
    out_dims[0] = std::get<3>(output_dims[0]);

    mv_preprocess_callback_args preproc_args;
    preproc_args.loop_decode = loop_decode;
    preproc_args.preproc_a   = scale_factor;
    preproc_args.preproc_b   = 0.0;

    std::string inp_dec_str;
    // For video input, set preprocessing callback for adding video decoder node
    if(inp_dims[3] == 1 && inp_dims[2] == 3 && input_file_name.size() > 4
       && ((input_file_name.substr(input_file_name.size() - 4, 4) == ".mp4")
           || (input_file_name.substr(input_file_name.size() - 4, 4) == ".m4v")))
    {
        inp_dec_str = use_vaapi ? ("1," + input_file_name + ":1") : ("1," + input_file_name + ":0");
        preproc_args.inp_string_decoder = inp_dec_str.c_str();
        SetPreProcessCallback(&preprocess_addnodes_callback_fn, &preproc_args);
        do_preprocess = 1;
        printf("OK:: SetPreProcessCallback\n");
    }
    else if(inp_dims[3] > 1 && inp_dims[2] == 3
            && (((input_file_name.size() > 6)
                 && (input_file_name.substr(input_file_name.size() - 6, 4) == ".mp4"
                     || input_file_name.substr(input_file_name.size() - 6, 4) == ".m4v"))
                || (input_file_name.substr(input_file_name.size() - 4, 4) == ".txt")))
    {
        inp_dec_str                     = std::to_string(inp_dims[3]) + "," + input_file_name;
        preproc_args.inp_string_decoder = inp_dec_str.c_str();
        SetPreProcessCallback(&preprocess_addnodes_callback_fn, &preproc_args);
        do_preprocess = 1;
        printf("OK:: SetPreProcessCallback\n");
    }

    status = mvCreateInferenceSession(&inf_session, install_folder.c_str(), mv_mem_type_host);
    if(status != MV_SUCCESS)
    {
        printf("ERROR: mvCreateInferenceSession returned failure\n");
        return -1;
    }

    inf_hdl = (mivid_handle)inf_session;
    // Get context from inference handle - required by ERROR_CHECK_STATUS macro in mvdeploy.h
    [[maybe_unused]] vx_context context = inf_hdl->context;

    // Create input tensor memory
    size_t input_size_in_bytes = 4 * inp_dims[0] * inp_dims[1] * inp_dims[2] * inp_dims[3];
    inp_mem                    = (float*)new char[input_size_in_bytes];
    size_t istride[4]          = {4,
                                  (size_t)4 * inp_dims[0],
                                  (size_t)4 * inp_dims[0] * inp_dims[1],
                                  (size_t)4 * inp_dims[0] * inp_dims[1] * inp_dims[2]};

    // Check if the input file is an image
    cv::Mat  img;
    cv::Mat  mat_scaled;
    cv::Mat* inp_img_mat = nullptr;
    if(!do_preprocess)
    {
#if ENABLE_OPENCV
        if(inp_dims[2] == 3 && input_file_name.size() > 4
           && (input_file_name.substr(input_file_name.size() - 4, 4) == ".png"
               || input_file_name.substr(input_file_name.size() - 4, 4) == ".jpg"
               || input_file_name.substr(input_file_name.size() - 4, 4) == ".PNG"
               || input_file_name.substr(input_file_name.size() - 4, 4) == ".JPG"))
        {
            for(size_t n = 0; n < inp_dims[3]; n++)
            {
                char img_file_name[1024];
                snprintf(img_file_name, sizeof(img_file_name), input_file_name.c_str(), (int)n);
                unsigned char* img_data;
                img         = cv::imread(img_file_name, cv::IMREAD_COLOR);
                img_data    = img.data;
                inp_img_mat = &img;
                if(!img.data || img.rows != (int)inp_dims[1] || img.cols != (int)inp_dims[0])
                {
                    cv::resize(img, mat_scaled, cv::Size(inp_dims[0], inp_dims[1]));
                    img_data    = mat_scaled.data;
                    inp_img_mat = &mat_scaled;
                }
                for(vx_size y = 0; y < inp_dims[1]; y++)
                {
                    unsigned char* src  = img_data + y * inp_dims[0] * 3;
                    float*         dstR = inp_mem + ((n * istride[3] + y * istride[1]) >> 2);
                    float*         dstG = dstR + (istride[2] >> 2);
                    float*         dstB = dstG + (istride[2] >> 2);
                    for(vx_size x = 0; x < inp_dims[0]; x++, src += 3)
                    {
                        *dstR++ = src[2] * scale_factor + add_factor;
                        *dstG++ = src[1] * scale_factor + add_factor;
                        *dstB++ = src[0] * scale_factor + add_factor;
                    }
                }
            }
        }
        else
#endif
        {
            FILE* fp = fopen(input_file_name.c_str(), "rb");
            if(!fp)
            {
                std::cerr << "ERROR: unable to open: " << input_file_name << std::endl;
                return -1;
            }
            for(size_t n = 0; n < inp_dims[3]; n++)
            {
                for(size_t c = 0; c < inp_dims[2]; c++)
                {
                    for(size_t y = 0; y < inp_dims[1]; y++)
                    {
                        float* ptrY
                            = inp_mem + ((n * istride[3] + c * istride[2] + y * istride[1]) >> 2);
                        vx_size num_read = fread(ptrY, sizeof(float), inp_dims[0], fp);
                        if(num_read != inp_dims[0])
                        {
                            std::cerr << "ERROR: reading from file less than expected # of bytes "
                                      << input_file_name << std::endl;
                            return -1;
                        }
                    }
                }
            }
            fclose(fp);
        }
        if((status = mvSetInputDataFromMemory(inf_session,
                                              0,
                                              (void*)inp_mem,
                                              input_size_in_bytes,
                                              mv_mem_type_host))
           != MV_SUCCESS)
        {
            printf("ERROR: mvSetInputDataFromMemory returned failure(%d)\n", status);
            return -1;
        }
    }

    size_t output_size_in_bytes = 4 * out_dims[0] * out_dims[1] * out_dims[2] * out_dims[3];
    out_mem                     = (float*)new char[output_size_in_bytes];
    std::vector<BBox> detected_bb;
    FILE*             fp = nullptr;

    if(output_file_name != "-")
    {
        fp = fopen(output_file_name.c_str(), "wb");
        if(!fp)
        {
            std::cerr << "ERROR: unable to open: " << output_file_name << std::endl;
            return -1;
        }
    }

    int64_t freq = clock_frequency();
    int64_t t0;
    int64_t t1;
    int64_t total_time = 0;
    int     fn;

    // Initialize postprocessing for object detection
    bool detect_bounding_boxes = (num_classes > 0);
    if(detect_bounding_boxes)
    {
        if((status = mv_postproc_init(inf_session,
                                      num_classes,
                                      13,
                                      bb_biases,
                                      10,
                                      conf_th,
                                      nms_th,
                                      inp_dims[0],
                                      inp_dims[1])))
        {
            printf("ERROR: mv_postproc_init failed with status(%d)\n", status);
            return -1;
        }
    }

    Visualize* p_visualize = nullptr;
    if(visualize)
    {
        p_visualize = new Visualize(0.2f);
    }

    ClassLabel cl[5];
    float      time_in_millisec;

    for(fn = 0; fn < end_frame || capture_till_eof; fn++)
    {
        t0     = clock_counter();
        status = mvRunInference(inf_session, &time_in_millisec, num_iterations);
        if(status == MV_ERROR_GRAPH_ABANDONED)
        {
            break;
        }
        if(status < 0)
        {
            printf("ERROR: mvRunInference terminated with status(%d)\n", status);
            return -1;
        }
        t1 = clock_counter();
        total_time += (t1 - t0);

        // Get output
        if((status = mvGetOutputData(inf_session, 0, (void*)out_mem, output_size_in_bytes))
           != MV_SUCCESS)
        {
            printf("ERROR: mvGetOutputData returned failure(%d)\n", status);
            return -1;
        }

        if(fp != nullptr)
        {
            fwrite(out_mem, sizeof(float), output_size_in_bytes >> 2, fp);
        }

        // Do object detection of output
        if(detect_bounding_boxes)
        {
            if((status = mv_postproc_getBB_detections(inf_session,
                                                      out_mem,
                                                      out_dims[3],
                                                      out_dims[2],
                                                      out_dims[1],
                                                      out_dims[0],
                                                      detected_bb)))
            {
                printf("ERROR: mv_postproc_getBB_detections returned status(%d)\n", status);
                return -1;
            }
        }

        if(argmax_topk > 0)
        {
            mv_postproc_argmax(out_mem,
                               (void*)cl,
                               argmax_topk,
                               out_dims[3],
                               out_dims[2],
                               out_dims[1],
                               out_dims[0]);
            for(int l = 0; l < argmax_topk; l++)
            {
                printf("Argmax topK: %d class:%d conf: %7.5f\n", l, cl[l].index, cl[l].probability);
            }
        }

        if(visualize && p_visualize)
        {
            inp_img = inf_hdl->inp_image;
            if(inp_img)
            {
                vx_uint32 width;
                vx_uint32 height;
                cv::Mat   img;

                if(inp_dims[3] == 4)
                {
                    img.create(inp_dims[1] * 2, inp_dims[0] * 2, CV_8UC3);
                }
                else if(inp_dims[3] == 8)
                {
                    img.create(inp_dims[1] * 2, inp_dims[0] * 4, CV_8UC3);
                }
                else if(inp_dims[3] == 16)
                {
                    img.create(inp_dims[1] * 4, inp_dims[0] * 4, CV_8UC3);
                }
                else
                {
                    img.create(inp_dims[1] * inp_dims[3], inp_dims[0], CV_8UC3);
                }

                vx_imagepatch_addressing_t addr{};
                ERROR_CHECK_STATUS(vxQueryImage(inp_img, VX_IMAGE_WIDTH, &width, sizeof(width)));
                ERROR_CHECK_STATUS(vxQueryImage(inp_img, VX_IMAGE_HEIGHT, &height, sizeof(height)));

                vx_rectangle_t rect_1 = {0, 0, width, height};
                vx_map_id      map_id;
                vx_uint8*      src = NULL;
                ERROR_CHECK_STATUS(vxMapImagePatch(inp_img,
                                                   &rect_1,
                                                   0,
                                                   &map_id,
                                                   &addr,
                                                   (void**)&src,
                                                   VX_READ_ONLY,
                                                   VX_MEMORY_TYPE_HOST,
                                                   VX_NOGAP_X));

                if(inp_dims[3] >= 4)
                {
                    vx_uint32 height1 = height / inp_dims[3];
                    // Copy images 0 and 1 to dest at loc (0,0) and (0,1)
                    for(vx_uint32 y = 0; y < height1; y++)
                    {
                        vx_uint8* p_dst  = (vx_uint8*)img.data + y * img.step;
                        vx_uint8* p_dst1 = p_dst + width * 3;
                        vx_uint8* p_src  = (vx_uint8*)src + y * addr.stride_y;
                        vx_uint8* p_src1 = (vx_uint8*)src + (y + height1) * addr.stride_y;
                        for(vx_uint32 x = 0; x < width; x++)
                        {
                            p_dst[0]  = p_src[2];
                            p_dst[1]  = p_src[1];
                            p_dst[2]  = p_src[0];
                            p_dst1[0] = p_src1[2];
                            p_dst1[1] = p_src1[1];
                            p_dst1[2] = p_src1[0];
                            p_dst += 3;
                            p_dst1 += 3;
                            p_src += 3;
                            p_src1 += 3;
                        }
                    }
                    // If num_images > 4, copy img 2 and 3 to dst at loc (0,2) and (0,3)
                    if(inp_dims[3] > 4)
                    {
                        for(vx_uint32 y = 0; y < height1; y++)
                        {
                            vx_uint8* p_dst  = (vx_uint8*)img.data + width * 6 + y * img.step;
                            vx_uint8* p_dst1 = p_dst + width * 3;
                            vx_uint8* p_src  = (vx_uint8*)src + (y + height1 * 2) * addr.stride_y;
                            vx_uint8* p_src1 = (vx_uint8*)src + (y + height1 * 3) * addr.stride_y;
                            for(vx_uint32 x = 0; x < width; x++)
                            {
                                p_dst[0]  = p_src[2];
                                p_dst[1]  = p_src[1];
                                p_dst[2]  = p_src[0];
                                p_dst1[0] = p_src1[2];
                                p_dst1[1] = p_src1[1];
                                p_dst1[2] = p_src1[0];
                                p_dst += 3;
                                p_dst1 += 3;
                                p_src += 3;
                                p_src1 += 3;
                            }
                        }
                    }
                    // Copy images 2 & 3 / 4 & 5 to dest at loc (1,0) and (1,1)
                    for(vx_uint32 y = 0; y < height1; y++)
                    {
                        vx_uint8* p_dst  = (vx_uint8*)img.data + (y + height1) * img.step;
                        vx_uint8* p_dst1 = p_dst + width * 3;
                        vx_uint8 *p_src, *p_src1;
                        if(inp_dims[3] == 4)
                        {
                            p_src  = (vx_uint8*)src + (y + height1 * 2) * addr.stride_y;
                            p_src1 = (vx_uint8*)src + (y + height1 * 3) * addr.stride_y;
                        }
                        else
                        {
                            p_src  = (vx_uint8*)src + (y + height1 * 4) * addr.stride_y;
                            p_src1 = (vx_uint8*)src + (y + height1 * 5) * addr.stride_y;
                        }
                        for(vx_uint32 x = 0; x < width; x++)
                        {
                            p_dst[0]  = p_src[2];
                            p_dst[1]  = p_src[1];
                            p_dst[2]  = p_src[0];
                            p_dst1[0] = p_src1[2];
                            p_dst1[1] = p_src1[1];
                            p_dst1[2] = p_src1[0];
                            p_dst += 3;
                            p_dst1 += 3;
                            p_src += 3;
                            p_src1 += 3;
                        }
                    }
                    // If num_images > 4, copy img 6 and 7 to dst at loc (1,2) and (1,3)
                    if(inp_dims[3] > 4)
                    {
                        for(vx_uint32 y = 0; y < height1; y++)
                        {
                            vx_uint8* p_dst
                                = (vx_uint8*)img.data + width * 6 + (y + height1) * img.step;
                            vx_uint8* p_dst1 = p_dst + width * 3;
                            vx_uint8* p_src  = (vx_uint8*)src + (y + height1 * 6) * addr.stride_y;
                            vx_uint8* p_src1 = (vx_uint8*)src + (y + height1 * 7) * addr.stride_y;
                            for(vx_uint32 x = 0; x < width; x++)
                            {
                                p_dst[0]  = p_src[2];
                                p_dst[1]  = p_src[1];
                                p_dst[2]  = p_src[0];
                                p_dst1[0] = p_src1[2];
                                p_dst1[1] = p_src1[1];
                                p_dst1[2] = p_src1[0];
                                p_dst += 3;
                                p_dst1 += 3;
                                p_src += 3;
                                p_src1 += 3;
                            }
                        }
                    }
                    // The following code is for batch 12 and 16
                    if(inp_dims[3] > 8)
                    {
                        // Copy images 8, 9, 10 and 11 to dest at loc (2,0), (2,1), (2,2) and (2,3)
                        for(vx_uint32 y = 0; y < height1; y++)
                        {
                            vx_uint8* p_dst  = (vx_uint8*)img.data + (y + height1 * 2) * img.step;
                            vx_uint8* p_dst1 = p_dst + width * 3;
                            vx_uint8* p_dst2 = p_dst1 + width * 3;
                            vx_uint8* p_dst3 = p_dst2 + width * 3;
                            vx_uint8* p_src  = (vx_uint8*)src + (y + height1 * 8) * addr.stride_y;
                            vx_uint8* p_src1 = (vx_uint8*)src + (y + height1 * 9) * addr.stride_y;
                            vx_uint8* p_src2 = (vx_uint8*)src + (y + height1 * 10) * addr.stride_y;
                            vx_uint8* p_src3 = (vx_uint8*)src + (y + height1 * 11) * addr.stride_y;
                            for(vx_uint32 x = 0; x < width; x++)
                            {
                                p_dst[0]  = p_src[2];
                                p_dst[1]  = p_src[1];
                                p_dst[2]  = p_src[0];
                                p_dst1[0] = p_src1[2];
                                p_dst1[1] = p_src1[1];
                                p_dst1[2] = p_src1[0];
                                p_dst2[0] = p_src2[2];
                                p_dst2[1] = p_src2[1];
                                p_dst2[2] = p_src2[0];
                                p_dst3[0] = p_src3[2];
                                p_dst3[1] = p_src3[1];
                                p_dst3[2] = p_src3[0];
                                p_dst += 3;
                                p_dst1 += 3;
                                p_dst2 += 3;
                                p_dst3 += 3;
                                p_src += 3;
                                p_src1 += 3;
                                p_src2 += 3;
                                p_src3 += 3;
                            }
                        }
                    }
                    if(inp_dims[3] > 12)
                    {
                        // Copy images 12, 13, 14 and 15 to dest at loc (3,0), (3,1), (3,2) and (3,3)
                        for(vx_uint32 y = 0; y < height1; y++)
                        {
                            vx_uint8* p_dst  = (vx_uint8*)img.data + (y + height1 * 3) * img.step;
                            vx_uint8* p_dst1 = p_dst + width * 3;
                            vx_uint8* p_dst2 = p_dst1 + width * 3;
                            vx_uint8* p_dst3 = p_dst2 + width * 3;
                            vx_uint8* p_src  = (vx_uint8*)src + (y + height1 * 12) * addr.stride_y;
                            vx_uint8* p_src1 = (vx_uint8*)src + (y + height1 * 13) * addr.stride_y;
                            vx_uint8* p_src2 = (vx_uint8*)src + (y + height1 * 14) * addr.stride_y;
                            vx_uint8* p_src3 = (vx_uint8*)src + (y + height1 * 15) * addr.stride_y;
                            for(vx_uint32 x = 0; x < width; x++)
                            {
                                p_dst[0]  = p_src[2];
                                p_dst[1]  = p_src[1];
                                p_dst[2]  = p_src[0];
                                p_dst1[0] = p_src1[2];
                                p_dst1[1] = p_src1[1];
                                p_dst1[2] = p_src1[0];
                                p_dst2[0] = p_src2[2];
                                p_dst2[1] = p_src2[1];
                                p_dst2[2] = p_src2[0];
                                p_dst3[0] = p_src3[2];
                                p_dst3[1] = p_src3[1];
                                p_dst3[2] = p_src3[0];
                                p_dst += 3;
                                p_dst1 += 3;
                                p_dst2 += 3;
                                p_dst3 += 3;
                                p_src += 3;
                                p_src1 += 3;
                                p_src2 += 3;
                                p_src3 += 3;
                            }
                        }
                    }
                }
                else
                {
                    for(vx_uint32 y = 0; y < height; y++)
                    {
                        vx_uint8* p_dst = (vx_uint8*)img.data + y * img.step;
                        vx_uint8* p_src = (vx_uint8*)src + y * addr.stride_y;
                        for(vx_uint32 x = 0; x < width; x++)
                        {
                            p_dst[0] = p_src[2];
                            p_dst[1] = p_src[1];
                            p_dst[2] = p_src[0];
                            p_dst += 3;
                            p_src += 3;
                        }
                    }
                }

                ERROR_CHECK_STATUS(vxUnmapImagePatch(inp_img, map_id));
                p_visualize->show(img, detected_bb, inp_dims[3]);
                if(cv::waitKey(1) >= 0)
                {
                    break;
                }
            }
            else if(inp_img_mat)
            {
                // Check if we read from file to cv::img
                p_visualize->show(*inp_img_mat, detected_bb);
                cv::waitKey(0);
            }
        }
    }

    if(fp)
    {
        fclose(fp);
    }
    if(fn)
    {
        float time_in_ms = (float)total_time * 1000.0f / (float)freq / (float)fn;
        std::cout << "OK: mvRunInference() took " << time_in_ms << " msec (average over " << fn
                  << " iterations)" << std::endl;
    }
    if(p_visualize)
    {
        delete p_visualize;
    }

    if(detect_bounding_boxes)
    {
        mv_postproc_shutdown(inf_session);
    }
    // Release Inference
    mvReleaseInferenceSession(inf_session);
    std::cout << "OK: Inference Deploy Successful" << std::endl;
    // Delete resources
    if(inp_mem)
    {
        delete[] inp_mem;
    }
    if(out_mem)
    {
        delete[] out_mem;
    }
    mvShutdown();

    return 0;
}
