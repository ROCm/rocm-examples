/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "roc_video_dec.h"
#include "video_demuxer.h"
#include "video_post_process.h"
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <libgen.h>
#include <mutex>
#include <queue>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "CmdParser/cmdparser.hpp"
#include "example_utils.hpp"
#include "rocdecode_utils.hpp"

std::vector<std::string> st_output_format_name
    = {"native", "bgr", "bgr48", "rgb", "rgb48", "bgra", "bgra64", "rgba", "rgba64"};

constexpr int           frame_buffers_size = 2;
std::mutex              mutex;
std::condition_variable cv;
std::queue<int>         frame_indices_q;
uint8_t*                frame_buffers[frame_buffers_size] = {0};

void color_space_conversion_thread(std::atomic<bool>&  continue_processing,
                                   bool                convert_to_rgb,
                                   Dim*                p_resize_dim,
                                   OutputSurfaceInfo** surf_info,
                                   OutputSurfaceInfo** res_surf_info,
                                   OutputFormatEnum    e_output_format,
                                   uint8_t*            p_rgb_dev_mem,
                                   uint8_t*            p_resize_dev_mem,
                                   bool                dump_output_frames,
                                   std::string&        output_file_path,
                                   RocVideoDecoder&    viddec,
                                   VideoPostProcess&   post_proc,
                                   MD5Generator*       md5_gen_handle,
                                   bool                b_generate_md5,
                                   int                 device_id,
                                   hipStream_t         hip_stream)
{

    size_t     rgb_image_size, resize_image_size;
    hipError_t hip_status = hipSuccess;
    int        current_frame_index;
    uint8_t*   frame;

    HIP_CHECK(hipSetDevice(device_id));
    while(continue_processing || !frame_indices_q.empty())
    {
        OutputSurfaceInfo* p_surf_info;
        uint8_t*           out_frame;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return !frame_indices_q.empty() || !continue_processing; });
            if(!continue_processing && frame_indices_q.empty())
            {
                break;
            }
            p_surf_info         = *surf_info;
            current_frame_index = frame_indices_q.front();
            frame               = frame_buffers[current_frame_index];
            out_frame           = frame;
        }
        if(p_resize_dim->w && p_resize_dim->h && *res_surf_info)
        {
            if(((*surf_info)->output_width != static_cast<uint32_t>(p_resize_dim->w))
               || ((*surf_info)->output_height != static_cast<uint32_t>(p_resize_dim->h)))
            {
                resize_image_size = p_resize_dim->w * (p_resize_dim->h + (p_resize_dim->h >> 1))
                                    * (*surf_info)->bytes_per_pixel;
                if(p_resize_dev_mem == nullptr && resize_image_size > 0)
                {
                    hip_status = hipMalloc(&p_resize_dev_mem, resize_image_size);
                    if(hip_status != hipSuccess)
                    {
                        std::cerr << "ERROR: hipMalloc failed to allocate the device memory for "
                                     "the output!"
                                  << hip_status << std::endl;
                        return;
                    }
                }
                if((*surf_info)->bytes_per_pixel == 2)
                {
                    ResizeP016(p_resize_dev_mem,
                               p_resize_dim->w * 2,
                               p_resize_dim->w,
                               p_resize_dim->h,
                               frame,
                               (*surf_info)->output_pitch,
                               (*surf_info)->output_width,
                               (*surf_info)->output_height,
                               (frame + (*surf_info)->output_vstride * (*surf_info)->output_pitch),
                               nullptr,
                               hip_stream);
                }
                else
                {
                    ResizeNv12(p_resize_dev_mem,
                               p_resize_dim->w,
                               p_resize_dim->w,
                               p_resize_dim->h,
                               frame,
                               (*surf_info)->output_pitch,
                               (*surf_info)->output_width,
                               (*surf_info)->output_height,
                               (frame + (*surf_info)->output_vstride * (*surf_info)->output_pitch),
                               nullptr,
                               hip_stream);
                }
                (*res_surf_info)->output_width   = p_resize_dim->w;
                (*res_surf_info)->output_height  = p_resize_dim->h;
                (*res_surf_info)->output_pitch   = p_resize_dim->w * (*surf_info)->bytes_per_pixel;
                (*res_surf_info)->output_vstride = p_resize_dim->h;
                (*res_surf_info)->output_surface_size_in_bytes
                    = (*res_surf_info)->output_pitch * (p_resize_dim->h + (p_resize_dim->h >> 1));
                (*res_surf_info)->mem_type = OUT_SURFACE_MEM_DEV_COPIED;
                p_surf_info                = *res_surf_info;
                out_frame                  = p_resize_dev_mem;
            }
        }

        if(convert_to_rgb)
        {
            uint32_t rgb_stride = post_proc.GetRgbStride(e_output_format, p_surf_info);
            rgb_image_size      = p_surf_info->output_height * rgb_stride;
            if(p_rgb_dev_mem == nullptr)
            {
                hip_status = hipMalloc(&p_rgb_dev_mem, rgb_image_size);
                if(hip_status != hipSuccess)
                {
                    std::cerr
                        << "ERROR: hipMalloc failed to allocate the device memory for the output!"
                        << hip_status << std::endl;
                    return;
                }
            }
            post_proc.ColorConvertYUV2RGB(out_frame,
                                          p_surf_info,
                                          p_rgb_dev_mem,
                                          e_output_format,
                                          hip_stream);
        }
        if(dump_output_frames)
        {
            if(convert_to_rgb)
            {
                viddec.SaveFrameToFile(output_file_path,
                                       p_rgb_dev_mem,
                                       p_surf_info,
                                       rgb_image_size);
            }
            else
            {
                viddec.SaveFrameToFile(output_file_path, out_frame, p_surf_info);
            }
        }
        if(b_generate_md5)
        {
            if(convert_to_rgb)
            {
                md5_gen_handle->UpdateMd5ForDataBuffer(p_rgb_dev_mem, rgb_image_size);
            }
            else
            {
                md5_gen_handle->UpdateMd5ForFrame(frame, p_surf_info);
            }
        }

        {
            std::unique_lock<std::mutex> lock(mutex);
            frame_indices_q.pop();
        }

        cv.notify_one();
    }
}

int main(int argc, char** argv)
{
    // Parse command-line arguments
    cli::Parser parser(argc, argv);
    parser.set_optional<std::string>("i", "input", "", "Input file path (required)");
    parser.set_optional<std::string>("o",
                                     "output",
                                     "",
                                     "Output file path - dumps output if requested");
    parser.set_optional<int>("d", "device", 0, "GPU device ID");
    parser.set_optional<std::string>(
        "of",
        "output_format",
        "native",
        "Output format (native, bgr, bgr48, rgb, rgb48, bgra, bgra64, rgba, rgba64)");
    parser.set_optional<std::string>("resize",
                                     "resize",
                                     "",
                                     "Resize dimensions WxH (e.g., 1920x1080)");
    parser.set_optional<std::string>("crop",
                                     "crop",
                                     "",
                                     "Crop rectangle for output (left,top,right,bottom)");
    parser.set_optional<int>("disp_delay",
                             "disp_delay",
                             1,
                             "Number of frames to be delayed for display");
    parser.set_optional<bool>("md5", "md5", false, "Generate MD5 message digest");
    parser.set_optional<std::string>("md5_check",
                                     "md5_check",
                                     "",
                                     "MD5 file path - generate MD5 and compare to reference");
    parser.run_and_exit_if_error();

    // Get parameters
    std::string input_file_path   = parser.get<std::string>("i");
    std::string output_file_path  = parser.get<std::string>("o");
    int         device_id         = parser.get<int>("d");
    std::string output_format_str = parser.get<std::string>("of");
    std::string resize_str        = parser.get<std::string>("resize");
    std::string crop_str          = parser.get<std::string>("crop");
    int         disp_delay        = parser.get<int>("disp_delay");
    bool        b_generate_md5    = parser.get<bool>("md5");
    std::string md5_file_path     = parser.get<std::string>("md5_check");

    if(input_file_path.empty())
    {
        std::cerr << "Error: Input file path is required. Use -i option." << std::endl;
        return 1;
    }

    bool dump_output_frames = !output_file_path.empty();
    bool b_md5_check        = !md5_file_path.empty();
    if(b_md5_check)
    {
        b_generate_md5 = true;
    }
    bool b_extract_sei_messages = false;

    Rect  crop_rect   = {};
    Rect* p_crop_rect = nullptr;
    if(!crop_str.empty())
    {
        if(4
           != sscanf(crop_str.c_str(),
                     "%d,%d,%d,%d",
                     &crop_rect.left,
                     &crop_rect.top,
                     &crop_rect.right,
                     &crop_rect.bottom))
        {
            std::cerr << "Error: Invalid crop rectangle format. Use: left,top,right,bottom"
                      << std::endl;
            return 1;
        }
        if((crop_rect.right - crop_rect.left) % 2 == 1
           || (crop_rect.bottom - crop_rect.top) % 2 == 1)
        {
            std::cout << "output crop rectangle must have width and height of even numbers"
                      << std::endl;
            return 1;
        }
        p_crop_rect = &crop_rect;
    }

    Dim resize_dim = {};
    if(!resize_str.empty())
    {
        if(2 != sscanf(resize_str.c_str(), "%dx%d", &resize_dim.w, &resize_dim.h))
        {
            std::cerr << "Error: Invalid resize format. Use: WxH (e.g., 1920x1080)" << std::endl;
            return 1;
        }
        if(resize_dim.w % 2 == 1 || resize_dim.h % 2 == 1)
        {
            std::cout << "Resizing dimensions must have width and height of even numbers"
                      << std::endl;
            return 1;
        }
    }

    OutputFormatEnum e_output_format = native;
    auto             it
        = std::find(st_output_format_name.begin(), st_output_format_name.end(), output_format_str);
    if(it == st_output_format_name.end())
    {
        std::cerr << "Error: Invalid output format. Valid options: native, bgr, bgr48, rgb, rgb48, "
                     "bgra, bgra64, rgba, rgba64"
                  << std::endl;
        return 1;
    }
    e_output_format = (OutputFormatEnum)(it - st_output_format_name.begin());

    hipError_t              hip_status          = hipSuccess;
    uint8_t*                p_rgb_dev_mem       = nullptr;
    uint8_t*                p_resize_dev_mem    = nullptr;
    OutputSurfaceMemoryType mem_type            = OUT_SURFACE_MEM_DEV_INTERNAL;
    int                     current_frame_index = 0;
    hipStream_t             hip_stream_dec      = 0;
    hipStream_t             hip_stream_csc      = 0;

    try
    {
        VideoDemuxer     demuxer(input_file_path.c_str());
        rocDecVideoCodec rocdec_codec_id = AVCodec2RocDecVideoCodec(demuxer.GetCodecID());
        RocVideoDecoder  viddec(device_id,
                               mem_type,
                               rocdec_codec_id,
                               false,
                               p_crop_rect,
                               b_extract_sei_messages,
                               disp_delay);
        if(!viddec.CodecSupported(device_id, rocdec_codec_id, demuxer.GetBitDepth()))
        {
            std::cerr << "GPU doesn't support codec!" << std::endl;
            return 0;
        }
        VideoPostProcess post_process;
        MD5Generator*    md5_generator = nullptr;

        std::string device_name, gcn_arch_name;
        int         pci_bus_id, pci_domain_id, pci_device_id;

        viddec.GetDeviceinfo(device_name, gcn_arch_name, pci_bus_id, pci_domain_id, pci_device_id);
        std::cout << "info: Using GPU device " << device_id << " " << device_name << "["
                  << gcn_arch_name << "] on PCI bus " << std::setfill('0') << std::setw(2)
                  << std::right << std::hex << pci_bus_id << ":" << std::setfill('0')
                  << std::setw(2) << std::right << std::hex << pci_domain_id << "." << pci_device_id
                  << std::dec << std::endl;
        std::cout << "info: decoding started, please wait!" << std::endl;
        HIP_CHECK(hipStreamCreate(&hip_stream_dec));
        HIP_CHECK(hipStreamCreate(&hip_stream_csc));

        if(b_generate_md5)
        {
            md5_generator = new MD5Generator();
            md5_generator->InitMd5();
        }

        int                n_video_bytes = 0, n_frames_returned = 0, n_frame = 0;
        uint8_t*           p_video = nullptr;
        uint8_t*           p_frame = nullptr;
        int64_t            pts     = 0;
        OutputSurfaceInfo* surf_info;
        OutputSurfaceInfo* resize_surf_info = nullptr;
        double             total_dec_time   = 0;
        bool               convert_to_rgb   = e_output_format != native;
        std::atomic<bool>  continue_processing(true);
        std::thread        color_space_conversion_thread_obj(color_space_conversion_thread,
                                                      std::ref(continue_processing),
                                                      std::ref(convert_to_rgb),
                                                      &resize_dim,
                                                      &surf_info,
                                                      &resize_surf_info,
                                                      std::ref(e_output_format),
                                                      std::ref(p_rgb_dev_mem),
                                                      std::ref(p_resize_dev_mem),
                                                      std::ref(dump_output_frames),
                                                      std::ref(output_file_path),
                                                      std::ref(viddec),
                                                      std::ref(post_process),
                                                      md5_generator,
                                                      b_generate_md5,
                                                      device_id,
                                                      hip_stream_csc);

        auto start_time = std::chrono::high_resolution_clock::now();
        do
        {
            demuxer.Demux(&p_video, &n_video_bytes, &pts);
            n_frames_returned = viddec.DecodeFrame(p_video, n_video_bytes, 0, pts);
            if(!n_frame && !viddec.GetOutputSurfaceInfo(&surf_info))
            {
                std::cerr << "Error: Failed to get Output Image Info!" << std::endl;
                break;
            }
            if(resize_dim.w && resize_dim.h && !resize_surf_info)
            {
                resize_surf_info = new OutputSurfaceInfo;
                memcpy(resize_surf_info, surf_info, sizeof(OutputSurfaceInfo));
            }

            for(int i = 0; i < n_frames_returned; i++)
            {
                p_frame = viddec.GetFrame(&pts);
                if(frame_buffers[0] == nullptr)
                {
                    for(int i = 0; i < frame_buffers_size; i++)
                    {
                        HIP_CHECK(
                            hipMalloc(&frame_buffers[i], surf_info->output_surface_size_in_bytes));
                    }
                }

                {
                    std::unique_lock<std::mutex> lock(mutex);
                    cv.wait(lock, [&] { return frame_indices_q.size() < frame_buffers_size; });
                    HIP_CHECK(hipMemcpyDtoDAsync(frame_buffers[current_frame_index],
                                                 p_frame,
                                                 surf_info->output_surface_size_in_bytes,
                                                 hip_stream_dec));
                    HIP_CHECK(hipStreamSynchronize(hip_stream_dec));
                    frame_indices_q.push(current_frame_index);
                }

                viddec.ReleaseFrame(pts);
                current_frame_index = (current_frame_index + 1) % frame_buffers_size;
                cv.notify_one();
            }

            n_frame += n_frames_returned;
        }
        while(n_video_bytes);

        {
            std::unique_lock<std::mutex> lock(mutex);
            continue_processing = false;
        }

        cv.notify_one();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto time_per_frame
            = std::chrono::duration<double, std::milli>(end_time - start_time).count();
        total_dec_time += time_per_frame;

        color_space_conversion_thread_obj.join();

        if(p_rgb_dev_mem != nullptr)
        {
            hip_status = hipFree(p_rgb_dev_mem);
            if(hip_status != hipSuccess)
            {
                std::cout << "ERROR: hipFree failed! (" << hip_status << ")" << std::endl;
                return -1;
            }
        }
        for(int i = 0; i < frame_buffers_size; i++)
        {
            hip_status = hipFree(frame_buffers[i]);
            if(hip_status != hipSuccess)
            {
                std::cout << "ERROR: hipFree failed! (" << hip_status << ")" << std::endl;
            }
        }
        if(hip_stream_dec)
        {
            HIP_CHECK(hipStreamDestroy(hip_stream_dec));
        }
        if(hip_stream_csc)
        {
            HIP_CHECK(hipStreamDestroy(hip_stream_csc));
        }

        std::cout << "info: Total frame decoded: " << n_frame << std::endl;
        if(!dump_output_frames)
        {
            std::string info_message = "info: avg decoding time per frame (ms): ";
            if(convert_to_rgb)
            {
                info_message = "info: avg decoding and post processing time per frame (ms): ";
            }
            std::cout << info_message << total_dec_time / n_frame << std::endl;
            std::cout << "info: avg FPS: " << (n_frame / total_dec_time) * 1000 << std::endl;
        }
        if(resize_surf_info != nullptr)
        {
            delete resize_surf_info;
        }
        if(b_generate_md5)
        {
            uint8_t* digest;
            md5_generator->FinalizeMd5(&digest);
            std::cout << "MD5 message digest: ";
            for(int i = 0; i < 16; i++)
            {
                std::cout << std::setfill('0') << std::setw(2) << std::hex
                          << static_cast<int>(digest[i]);
            }
            std::cout << std::endl;
            if(b_md5_check)
            {
                std::string   ref_md5_string(33, 0);
                uint8_t       ref_md5[16];
                std::ifstream ref_md5_file(md5_file_path.c_str(), std::ios::in);
                if(!ref_md5_file)
                {
                    std::cerr << "Failed to open MD5 file." << std::endl;
                    return 1;
                }
                ref_md5_file.getline(ref_md5_string.data(), ref_md5_string.length());
                if(!ref_md5_file)
                {
                    std::cerr << "Failed to read MD5 digest string." << std::endl;
                    return 1;
                }
                for(int i = 0; i < 16; i++)
                {
                    std::string part = ref_md5_string.substr(i * 2, 2);
                    ref_md5[i]       = std::stoi(part, nullptr, 16);
                }
                if(memcmp(digest, ref_md5, 16) == 0)
                {
                    std::cout << "MD5 digest matches the reference MD5 digest: ";
                }
                else
                {
                    std::cout << "MD5 digest does not match the reference MD5 digest: ";
                }
                std::cout << ref_md5_string.c_str() << std::endl;
                ref_md5_file.close();
            }
            delete md5_generator;
        }
    }
    catch(const std::exception& ex)
    {
        std::cout << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
