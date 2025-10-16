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

#include "CmdParser/cmdparser.hpp"

#include <chrono>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <libgen.h>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>
#if __cplusplus >= 201703L && __has_include(<filesystem>)
    #include <filesystem>
#else
    #include <experimental/filesystem>
#endif
#include "roc_video_dec.h"
#include "video_demuxer.h"

#include "rocdecode_utils.hpp"

typedef struct
{
    std::string             in_file;
    std::string             out_file;
    bool                    b_force_zero_latency;
    bool                    b_extract_sei_messages;
    bool                    b_flush_last_frames;
    Rect                    crop_rect;
    Rect*                   p_crop_rect;
    int                     dump_output_frames;
    OutputSurfaceMemoryType mem_type;
    int                     disp_delay;
} file_info;

void parse_file_list(const std::string& file_list_path, std::deque<file_info>& multi_file_data)
{
    std::ifstream filestream(file_list_path);
    std::string   line;
    char          param[256];
    char          value[256];
    int           file_idx = 0;
    file_info     file_data;

    while(std::getline(filestream, line))
    {
        const char* str = line.c_str();
        if(sscanf(str, "%s %s", param, value) != 2)
        {
            continue;
        }

        if(!strcmp(param, "infile"))
        {
            if(file_idx > 0)
            {
                multi_file_data.push_back(file_data);
            }
            file_data.in_file = value;
            file_idx++;
            file_data.b_force_zero_latency   = false;
            file_data.b_extract_sei_messages = false;
            file_data.b_flush_last_frames    = true;
            file_data.dump_output_frames     = 0;
            file_data.crop_rect              = {};
            file_data.p_crop_rect            = nullptr;
            file_data.mem_type               = OUT_SURFACE_MEM_DEV_INTERNAL;
            file_data.disp_delay             = 1;
        }
        else if(!strcmp(param, "outfile"))
        {
            file_data.out_file           = value;
            file_data.dump_output_frames = 1;
        }
        else if(!strcmp(param, "z"))
        {
            file_data.b_force_zero_latency = atoi(value) ? true : false;
        }
        else if(!strcmp(param, "sei"))
        {
            file_data.b_extract_sei_messages = atoi(value) ? true : false;
        }
        else if(!strcmp(param, "flush"))
        {
            file_data.b_flush_last_frames = atoi(value) ? true : false;
        }
        else if(!strcmp(param, "crop"))
        {
            if(sscanf(value,
                      "%d,%d,%d,%d",
                      &file_data.crop_rect.left,
                      &file_data.crop_rect.top,
                      &file_data.crop_rect.right,
                      &file_data.crop_rect.bottom)
               == 4)
            {
                if((file_data.crop_rect.right - file_data.crop_rect.left) % 2 == 1
                   || (file_data.crop_rect.bottom - file_data.crop_rect.top) % 2 == 1)
                {
                    std::cerr << "Cropping rect must have width and height of even numbers"
                              << std::endl;
                    exit(1);
                }
                file_data.p_crop_rect = &file_data.crop_rect;
            }
        }
        else if(!strcmp(param, "m"))
        {
            file_data.mem_type = static_cast<OutputSurfaceMemoryType>(atoi(value));
        }
        else if(!strcmp(param, "disp_delay"))
        {
            file_data.disp_delay = atoi(value);
        }
    }
    if(file_idx > 0)
    {
        multi_file_data.push_back(file_data);
    }
}

void configure_parser(cli::Parser& parser)
{
    parser.set_required<std::string>("i",
                                     "input",
                                     "Input file list (text file containing all files to decode)");
    parser.set_optional<int>("d",
                             "device",
                             0,
                             "GPU device ID (0 for the first device, 1 for the second, etc.)");
    parser.set_optional<bool>(
        "use_reconfigure",
        "use_reconfigure",
        true,
        "Use reconfigure API for decoding multiple files (only resolution changes supported)");
}

int main(int argc, char** argv)
{
    // Parse command-line arguments
    cli::Parser parser(argc, argv);
    configure_parser(parser);
    parser.run_and_exit_if_error();

    // Get arguments
    std::string file_list_path  = parser.get<std::string>("i");
    int         device_id       = parser.get<int>("d");
    bool        use_reconfigure = parser.get<bool>("use_reconfigure");

    std::deque<file_info> multi_file_data;
    parse_file_list(file_list_path, multi_file_data);

    RocVideoDecoder*          viddec               = nullptr;
    ReconfigParams            reconfig_params      = {};
    reconfig_dump_file_struct reconfig_user_struct = {};

    try
    {
        while(!multi_file_data.empty())
        {
            file_info file_data = multi_file_data.front();
            multi_file_data.pop_front();
            VideoDemuxer     demuxer(file_data.in_file.c_str());
            rocDecVideoCodec rocdec_codec_id = AVCodec2RocDecVideoCodec(demuxer.GetCodecID());

            if(file_data.b_flush_last_frames && file_data.dump_output_frames)
            {
                reconfig_params.p_fn_reconfigure_flush     = reconfigure_flush_callback;
                reconfig_user_struct.b_dump_frames_to_file = file_data.dump_output_frames;
                reconfig_user_struct.output_file_name      = file_data.out_file;
                reconfig_params.reconfig_flush_mode        = RECONFIG_FLUSH_MODE_DUMP_TO_FILE;
                reconfig_params.p_reconfig_user_struct     = &reconfig_user_struct;
            }

            if(use_reconfigure)
            {
                if(!viddec)
                {
                    viddec = new RocVideoDecoder(device_id,
                                                 file_data.mem_type,
                                                 rocdec_codec_id,
                                                 file_data.b_force_zero_latency,
                                                 file_data.p_crop_rect,
                                                 file_data.b_extract_sei_messages,
                                                 file_data.disp_delay);
                }
            }
            else
            {
                viddec = new RocVideoDecoder(device_id,
                                             file_data.mem_type,
                                             rocdec_codec_id,
                                             file_data.b_force_zero_latency,
                                             file_data.p_crop_rect,
                                             file_data.b_extract_sei_messages,
                                             file_data.disp_delay);
            }

            if(!viddec->CodecSupported(device_id, rocdec_codec_id, demuxer.GetBitDepth()))
            {
                std::cerr << "Codec not supported on GPU, skipping this file!" << std::endl;
                continue;
            }

            if(viddec && file_data.b_flush_last_frames)
            {
                viddec->SetReconfigParams(&reconfig_params);
            }

            std::string device_name, gcn_arch_name;
            int         pci_bus_id, pci_domain_id, pci_device_id;

            std::size_t found_file = file_data.in_file.find_last_of('/');
            std::cout << "info: Input file: " << file_data.in_file.substr(found_file + 1)
                      << std::endl;
            viddec->GetDeviceinfo(device_name,
                                  gcn_arch_name,
                                  pci_bus_id,
                                  pci_domain_id,
                                  pci_device_id);
            std::cout << "info: Using GPU device " << device_id << " - " << device_name << "["
                      << gcn_arch_name << "] on PCI bus " << std::setfill('0') << std::setw(2)
                      << std::right << std::hex << pci_bus_id << ":" << std::setfill('0')
                      << std::setw(2) << std::right << std::hex << pci_domain_id << "."
                      << pci_device_id << std::dec << std::endl;
            std::cout << "info: decoding started, please wait!" << std::endl;

            int                n_video_bytes = 0, n_frame_returned = 0, n_frame = 0;
            uint8_t*           pvideo    = nullptr;
            int                pkg_flags = 0;
            uint8_t*           pframe    = nullptr;
            int64_t            pts       = 0;
            OutputSurfaceInfo* surf_info;
            double             total_dec_time = 0;

            do
            {
                auto start_time = std::chrono::high_resolution_clock::now();
                demuxer.Demux(&pvideo, &n_video_bytes, &pts);
                // Treat 0 bitstream size as end of stream indicator
                if(n_video_bytes == 0)
                {
                    pkg_flags |= ROCDEC_PKT_ENDOFSTREAM;
                }
                n_frame_returned = viddec->DecodeFrame(pvideo, n_video_bytes, pkg_flags, pts);
                auto end_time    = std::chrono::high_resolution_clock::now();
                auto time_per_frame
                    = std::chrono::duration<double, std::milli>(end_time - start_time).count();
                total_dec_time += time_per_frame;
                if(!n_frame && !viddec->GetOutputSurfaceInfo(&surf_info))
                {
                    std::cerr << "Error: Failed to get Output Surface Info!" << std::endl;
                    break;
                }
                for(int i = 0; i < n_frame_returned; i++)
                {
                    pframe = viddec->GetFrame(&pts);
                    if(file_data.dump_output_frames
                       && file_data.mem_type != OUT_SURFACE_MEM_NOT_MAPPED)
                    {
                        viddec->SaveFrameToFile(file_data.out_file, pframe, surf_info);
                    }
                    // release frame
                    viddec->ReleaseFrame(pts);
                }
                n_frame += n_frame_returned;
            }
            while(n_video_bytes);

            n_frame += viddec->GetNumOfFlushedFrames();
            std::cout << "info: Total frame decoded: " << n_frame << std::endl;
            if(!file_data.dump_output_frames)
            {
                std::cout << "info: avg decoding time per frame (ms): " << total_dec_time / n_frame
                          << std::endl;
                std::cout << "info: avg FPS: " << (n_frame / total_dec_time) * 1000 << std::endl;
            }
            else
            {
                if(file_data.mem_type == OUT_SURFACE_MEM_NOT_MAPPED)
                {
                    std::cout << "info: saving frames with -m 3 option is not supported!"
                              << std::endl;
                }
                else
                {
                    std::cout << "info: saved frames into " << file_data.out_file << std::endl;
                }
            }

            if(!use_reconfigure)
            {
                delete viddec;
                viddec = nullptr;
            }
            std::cout << "\n";
        }

        if(viddec)
        {
            delete viddec;
            viddec = nullptr;
        }
    }
    catch(const std::exception& ex)
    {
        std::cout << ex.what() << std::endl;
        exit(1);
    }

    return 0;
}
