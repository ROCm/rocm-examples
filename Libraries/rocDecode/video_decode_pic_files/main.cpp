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
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <libgen.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#if ENABLE_HOST_DECODE
    #include "ffmpeg_video_dec.h"
#endif

#include "CmdParser/cmdparser.hpp"
#include "example_utils.hpp"
#include "rocdecode_utils.hpp"

int main(int argc, char** argv)
{
    // Parse command-line arguments
    cli::Parser parser(argc, argv);
    parser.set_optional<std::vector<std::string>>("i",
                                                  "input",
                                                  {},
                                                  "Input picture files (required)");
    parser.set_optional<int>("codec",
                             "codec",
                             0,
                             "Codec type (0: HEVC, 1: AVC; 2: AV1; 3: VP9) - required");
    parser.set_optional<int>("l", "iterations", 1, "Number of iterations");
    parser.set_optional<std::string>("o",
                                     "output",
                                     "",
                                     "Output file path - dumps output if requested");
    parser.set_optional<int>("d", "device", 0, "GPU device ID");
    parser.set_optional<int>("backend",
                             "backend",
                             0,
                             "Backend (0 for GPU, 1 CPU-FFMpeg, 2 CPU-FFMpeg No threading)");
    parser.set_optional<int>(
        "f",
        "frames",
        0,
        "Number of decoded frames - specify the number of pictures to be decoded");
    parser.set_optional<bool>(
        "z",
        "zero_latency",
        false,
        "Force zero latency (decoded frames will be flushed out for display immediately)");
    parser.set_optional<int>("disp_delay",
                             "disp_delay",
                             1,
                             "Specify the number of frames to be delayed for display");
    parser.set_optional<bool>("md5",
                              "md5",
                              false,
                              "Generate MD5 message digest on the decoded YUV image sequence");
    parser.set_optional<std::string>(
        "md5_check",
        "md5_check",
        "",
        "MD5 file path - generate MD5 message digest and compare to reference");
    parser.set_optional<std::string>("crop",
                                     "crop",
                                     "",
                                     "Crop rectangle for output (left,top,right,bottom)");
    parser.set_optional<int>("m",
                             "mem_type",
                             0,
                             "Output surface memory type (0: DEV_INTERNAL, 1: DEV_COPIED, 2: "
                             "HOST_COPIED, 3: NOT_MAPPED)");
    parser.run_and_exit_if_error();

    // Get parameters
    std::vector<std::string> file_names           = parser.get<std::vector<std::string>>("i");
    int                      codec_type           = parser.get<int>("codec");
    int                      num_iterations       = parser.get<int>("l");
    std::string              output_file_path     = parser.get<std::string>("o");
    int                      device_id            = parser.get<int>("d");
    int                      backend              = parser.get<int>("backend");
    uint32_t                 num_decoded_frames   = parser.get<int>("f");
    bool                     b_force_zero_latency = parser.get<bool>("z");
    int                      disp_delay           = parser.get<int>("disp_delay");
    bool                     b_generate_md5       = parser.get<bool>("md5");
    std::string              md5_file_path        = parser.get<std::string>("md5_check");
    std::string              crop_str             = parser.get<std::string>("crop");
    OutputSurfaceMemoryType  mem_type = static_cast<OutputSurfaceMemoryType>(parser.get<int>("m"));

    if(file_names.empty())
    {
        std::cerr << "Error: Input files are required. Use -i option." << std::endl;
        return 1;
    }

    int  dump_output_frames = output_file_path.empty() ? 0 : 1;
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

    try
    {
        std::cout << "Total frame number = " << file_names.size() << std::endl;
        rocDecVideoCodec rocdec_codec_id;
        switch(codec_type)
        {
            case 0: rocdec_codec_id = rocDecVideoCodec_HEVC; break;
            case 1: rocdec_codec_id = rocDecVideoCodec_AVC; break;
            case 2: rocdec_codec_id = rocDecVideoCodec_AV1; break;
            case 3: rocdec_codec_id = rocDecVideoCodec_VP9; break;
            default: std::cerr << "Unsupported stream codec type." << std::endl; return 1;
        }

        RocVideoDecoder* viddec;
        if(!backend)
        {
            // gpu backend
            viddec = new RocVideoDecoder(device_id,
                                         mem_type,
                                         rocdec_codec_id,
                                         b_force_zero_latency,
                                         p_crop_rect,
                                         b_extract_sei_messages,
                                         disp_delay);
        }
        else
        {
#if ENABLE_HOST_DECODE
            std::cout << "info: RocDecode is using CPU backend!" << std::endl;
            if(mem_type == OUT_SURFACE_MEM_DEV_INTERNAL)
            {
                mem_type
                    = OUT_SURFACE_MEM_DEV_COPIED; // mem_type internal is not supported in this mode
            }
            if(backend == 1)
            {
                viddec = new FFMpegVideoDecoder(device_id,
                                                mem_type,
                                                rocdec_codec_id,
                                                b_force_zero_latency,
                                                p_crop_rect,
                                                b_extract_sei_messages,
                                                disp_delay);
            }
            else
            {
                viddec = new FFMpegVideoDecoder(device_id,
                                                mem_type,
                                                rocdec_codec_id,
                                                b_force_zero_latency,
                                                p_crop_rect,
                                                b_extract_sei_messages,
                                                disp_delay,
                                                true);
            }
#else
            std::cerr << "Error: CPU backend not enabled. Rebuild with ENABLE_HOST_DECODE=1"
                      << std::endl;
            return 1;
#endif
        }

        std::string device_name, gcn_arch_name;
        int         pci_bus_id, pci_domain_id, pci_device_id;

        viddec->GetDeviceinfo(device_name, gcn_arch_name, pci_bus_id, pci_domain_id, pci_device_id);
        std::cout << "info: Using GPU device " << device_id << " - " << device_name << "["
                  << gcn_arch_name << "] on PCI bus " << std::setfill('0') << std::setw(2)
                  << std::right << std::hex << pci_bus_id << ":" << std::setfill('0')
                  << std::setw(2) << std::right << std::hex << pci_domain_id << "." << pci_device_id
                  << std::dec << std::endl;
        std::cout << "info: decoding started, please wait!" << std::endl;

        int                  n_video_bytes = 0, n_frame_returned = 0, n_frame = 0;
        int                  n_pic_decoded = 0, decoded_pics = 0;
        std::vector<uint8_t> bitstream(5 * 1024 * 1024);
        int                  pkg_flags = 0;
        uint8_t*             pframe    = nullptr;
        int64_t              pts       = 0;
        OutputSurfaceInfo*   surf_info;
        double               total_dec_time = 0;
        MD5Generator*        md5_generator  = nullptr;

        // initialize reconfigure params
        ReconfigParams            reconfig_params      = {};
        reconfig_dump_file_struct reconfig_user_struct = {};
        reconfig_params.p_fn_reconfigure_flush         = reconfigure_flush_callback;
        reconfig_user_struct.b_dump_frames_to_file     = dump_output_frames;
        reconfig_user_struct.output_file_name          = output_file_path;
        reconfig_params.reconfig_flush_mode            = RECONFIG_FLUSH_MODE_NONE;
        if(dump_output_frames)
        {
            reconfig_params.reconfig_flush_mode |= RECONFIG_FLUSH_MODE_DUMP_TO_FILE;
        }
        if(b_generate_md5)
        {
            reconfig_params.reconfig_flush_mode |= RECONFIG_FLUSH_MODE_CALCULATE_MD5;
        }
        reconfig_params.p_reconfig_user_struct = &reconfig_user_struct;

        if(b_generate_md5)
        {
            md5_generator = new MD5Generator();
            md5_generator->InitMd5();
            reconfig_user_struct.md5_generator_handle = static_cast<void*>(md5_generator);
        }
        viddec->SetReconfigParams(&reconfig_params);

        for(int i = 0; i < num_iterations; i++)
        {
            int num_frames_decoded_in_loop = 0;
            pkg_flags                      = 0;
            for(const auto& file_name : file_names)
            {
                std::ifstream in_file(file_name, std::ios::binary);
                if(!in_file)
                {
                    std::cerr << "Error: Failed to open " << file_name << " for reading."
                              << std::endl;
                    return 1;
                }
                in_file.seekg(0, std::ios::end);
                n_video_bytes = in_file.tellg();
                if(static_cast<size_t>(n_video_bytes) > bitstream.size())
                {
                    bitstream.resize(n_video_bytes);
                }
                in_file.seekg(0, std::ios::beg);
                if(!in_file.read(reinterpret_cast<char*>(bitstream.data()), n_video_bytes))
                {
                    std::cerr << "Error: Failed to read " << file_name << "." << std::endl;
                    return 1;
                }
                in_file.close();

                auto start_time = std::chrono::high_resolution_clock::now();
                if(static_cast<size_t>(num_frames_decoded_in_loop + 1) == file_names.size())
                {
                    pkg_flags |= ROCDEC_PKT_ENDOFSTREAM;
                }
                n_frame_returned = viddec->DecodeFrame(bitstream.data(),
                                                       n_video_bytes,
                                                       pkg_flags,
                                                       pts,
                                                       &decoded_pics);
                num_frames_decoded_in_loop++;

                if(!n_frame && !viddec->GetOutputSurfaceInfo(&surf_info))
                {
                    std::cerr << "Error: Failed to get Output Surface Info!" << std::endl;
                    break;
                }
                for(int j = 0; j < n_frame_returned; j++)
                {
                    pframe = viddec->GetFrame(&pts);
                    if(b_generate_md5)
                    {
                        md5_generator->UpdateMd5ForFrame(pframe, surf_info);
                    }
                    if(dump_output_frames && mem_type != OUT_SURFACE_MEM_NOT_MAPPED)
                    {
                        viddec->SaveFrameToFile(output_file_path, pframe, surf_info);
                    }
                    viddec->ReleaseFrame(pts);
                }
                auto end_time = std::chrono::high_resolution_clock::now();
                auto time_per_decode
                    = std::chrono::duration<double, std::milli>(end_time - start_time).count();
                total_dec_time += time_per_decode;
                n_frame += n_frame_returned;
                n_pic_decoded += decoded_pics;
                if(num_decoded_frames && static_cast<uint32_t>(n_frame) >= num_decoded_frames)
                {
                    break;
                }
            }
        }
        n_frame += viddec->GetNumOfFlushedFrames();

        std::cout << "info: Total pictures decoded: " << n_pic_decoded << std::endl;
        std::cout << "info: Total frames output/displayed: " << n_frame << std::endl;
        if(!dump_output_frames)
        {
            std::cout << "info: avg decoding time per picture: " << total_dec_time / n_pic_decoded
                      << " ms" << std::endl;
            std::cout << "info: avg decode FPS: " << (n_pic_decoded / total_dec_time) * 1000
                      << std::endl;
            std::cout << "info: avg output/display time per frame: " << total_dec_time / n_frame
                      << " ms" << std::endl;
            std::cout << "info: avg output/display FPS: " << (n_frame / total_dec_time) * 1000
                      << std::endl;
        }
        else
        {
            if(mem_type == OUT_SURFACE_MEM_NOT_MAPPED)
            {
                std::cout << "info: saving frames with -m 3 option is not supported!" << std::endl;
            }
            else
            {
                std::cout << "info: saved frames into " << output_file_path << std::endl;
            }
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

        delete viddec;
    }
    catch(const std::exception& ex)
    {
        std::cout << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
