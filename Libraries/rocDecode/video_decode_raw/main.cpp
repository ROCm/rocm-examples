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

#include "rocdecode/roc_bitstream_reader.h"
#include "roc_video_dec.h"

#include "CmdParser/cmdparser.hpp"
#include "example_utils.hpp"

typedef enum reconfigure_flush_mode_enum {
    RECONFIG_FLUSH_MODE_NONE = 0x0,                      /**<  Just flush to get the frame count */
    RECONFIG_FLUSH_MODE_DUMP_TO_FILE = 0x1,              /**<  The remaining frames will be dumped to file in this mode */
} reconfigure_flush_mode;

// This struct is used by sample apps to dump last frames to file
typedef struct reconfig_dump_file_struct_t {
    bool b_dump_frames_to_file;
    std::string output_file_name;
} reconfig_dump_file_struct;

// Callback function to flush last frames and save it to file when reconfigure happens
inline int reconfigure_flush_callback(void *p_viddec_obj, uint32_t flush_mode, void *p_user_struct)
{
    int n_frames_flushed = 0;
    if ((p_viddec_obj == nullptr) || (p_user_struct == nullptr))
    {
        return n_frames_flushed;
    }

    RocVideoDecoder *viddec = static_cast<RocVideoDecoder *>(p_viddec_obj);
    OutputSurfaceInfo *surf_info;
    if (!viddec->GetOutputSurfaceInfo(&surf_info))
    {
        std::cerr << "Error: Failed to get Output Surface Info!" << std::endl;
        return n_frames_flushed;
    }

    uint8_t *pframe = nullptr;
    int64_t pts;
    while ((pframe = viddec->GetFrame(&pts)))
    {
        if (flush_mode != RECONFIG_FLUSH_MODE_NONE)
        {
            reconfig_dump_file_struct *p_dump_file_struct = static_cast<reconfig_dump_file_struct *>(p_user_struct);
            if (flush_mode & reconfigure_flush_mode::RECONFIG_FLUSH_MODE_DUMP_TO_FILE)
            {
                if (p_dump_file_struct->b_dump_frames_to_file)
                {
                    viddec->SaveFrameToFile(p_dump_file_struct->output_file_name, pframe, surf_info);
                }
            }
        }
        // release and flush frame
        viddec->ReleaseFrame(pts, true);
        n_frames_flushed++;
    }

    return n_frames_flushed;
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
    parser.set_optional<int>("d",
                             "device",
                             0,
                             "GPU device ID (0 for the first device, 1 for the second, etc.)");
    parser.set_optional<int>("f",
                             "frames",
                             0,
                             "Number of decoded frames - specify the number of pictures to be "
                             "decoded (0 = decode entire stream)");
    parser.set_optional<bool>(
        "z",
        "zero_latency",
        false,
        "Force zero latency (decoded frames will be flushed out for display immediately)");
    parser.set_optional<int>("disp_delay",
                             "display_delay",
                             1,
                             "Specify the number of frames to be delayed for display");
    parser.set_optional<bool>("sei", "extract_sei", false, "Extract SEI messages");
    parser.set_optional<std::string>("crop",
                                     "crop_rect",
                                     "",
                                     "Crop rectangle for output (format: left,top,right,bottom)");
    parser.set_optional<int>("m",
                             "memory_type",
                             0,
                             "Output surface memory type [0: DEV_INTERNAL, 1: DEV_COPIED, 2: "
                             "HOST_COPIED, 3: NOT_MAPPED]");
    parser.run_and_exit_if_error();

    // Get parameters
    std::string input_file_path = parser.get<std::string>("i");
    if(input_file_path.empty())
    {
        std::cerr << "Error: Input file path is required (-i option)" << std::endl;
        return 1;
    }

    std::string             output_file_path       = parser.get<std::string>("o");
    int                     dump_output_frames     = output_file_path.empty() ? 0 : 1;
    int                     device_id              = parser.get<int>("d");
    int                     disp_delay             = parser.get<int>("disp_delay");
    bool                    b_force_zero_latency   = parser.get<bool>("z");
    bool                    b_extract_sei_messages = parser.get<bool>("sei");
    uint32_t                num_decoded_frames     = parser.get<int>("f");
    OutputSurfaceMemoryType mem_type = static_cast<OutputSurfaceMemoryType>(parser.get<int>("m"));

    // Parse crop rectangle if provided
    Rect        crop_rect   = {};
    Rect*       p_crop_rect = nullptr;
    std::string crop_str    = parser.get<std::string>("crop");
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
            std::cerr << "Error: Output crop rectangle must have width and height of even numbers"
                      << std::endl;
            return 1;
        }
        p_crop_rect = &crop_rect;
    }

    try
    {
        std::size_t found_file = input_file_path.find_last_of('/');
        std::cout << "info: Input file: " << input_file_path.substr(found_file + 1) << std::endl;
        std::cout << "info: Using built-in bitstream reader" << std::endl;
        RocdecBitstreamReader bs_reader = nullptr;
        rocDecVideoCodec      rocdec_codec_id;
        int                   bit_depth;
        if(rocDecCreateBitstreamReader(&bs_reader, input_file_path.c_str()) != ROCDEC_SUCCESS)
        {
            std::cerr << "Failed to create the bitstream reader." << std::endl;
            return 1;
        }
        if(rocDecGetBitstreamCodecType(bs_reader, &rocdec_codec_id) != ROCDEC_SUCCESS)
        {
            std::cerr << "Failed to get stream codec type." << std::endl;
            return 1;
        }
        if(rocdec_codec_id >= rocDecVideoCodec_NumCodecs)
        {
            std::cerr
                << "Unsupported stream file type or codec type by the bitstream reader. Exiting."
                << std::endl;
            return 1;
        }
        if(rocDecGetBitstreamBitDepth(bs_reader, &bit_depth) != ROCDEC_SUCCESS)
        {
            std::cerr << "Failed to get stream bit depth." << std::endl;
            return 1;
        }

        RocVideoDecoder viddec(device_id,
                               mem_type,
                               rocdec_codec_id,
                               b_force_zero_latency,
                               p_crop_rect,
                               b_extract_sei_messages,
                               disp_delay);
        if(!viddec.CodecSupported(device_id, rocdec_codec_id, bit_depth))
        {
            std::cerr << "GPU doesn't support codec!" << std::endl;
            return 0;
        }
        std::string device_name, gcn_arch_name;
        int         pci_bus_id, pci_domain_id, pci_device_id;

        viddec.GetDeviceinfo(device_name, gcn_arch_name, pci_bus_id, pci_domain_id, pci_device_id);
        std::cout << "info: Using GPU device " << device_id << " - " << device_name << "["
                  << gcn_arch_name << "] on PCI bus " << std::setfill('0') << std::setw(2)
                  << std::right << std::hex << pci_bus_id << ":" << std::setfill('0')
                  << std::setw(2) << std::right << std::hex << pci_domain_id << "." << pci_device_id
                  << std::dec << std::endl;
        std::cout << "info: decoding started, please wait!" << std::endl;

        int                n_video_bytes = 0, n_frame_returned = 0, n_frame = 0;
        int                n_pic_decoded = 0, decoded_pics = 0;
        uint8_t*           pvideo    = nullptr;
        int                pkg_flags = 0;
        uint8_t*           pframe    = nullptr;
        int64_t            pts       = 0;
        OutputSurfaceInfo* surf_info;
        double             total_dec_time = 0;

        // Initialize reconfigure params
        ReconfigParams            reconfig_params      = {};
        reconfig_dump_file_struct reconfig_user_struct = {};
        reconfig_params.p_fn_reconfigure_flush         = reconfigure_flush_callback;
        reconfig_user_struct.b_dump_frames_to_file     = dump_output_frames;
        reconfig_user_struct.output_file_name          = output_file_path;
        if(dump_output_frames)
        {
            reconfig_params.reconfig_flush_mode |= RECONFIG_FLUSH_MODE_DUMP_TO_FILE;
        }
        else
        {
            reconfig_params.reconfig_flush_mode = RECONFIG_FLUSH_MODE_NONE;
        }
        reconfig_params.p_reconfig_user_struct = &reconfig_user_struct;

        viddec.SetReconfigParams(&reconfig_params);

        do
        {
            auto start_time = std::chrono::high_resolution_clock::now();
            if(rocDecGetBitstreamPicData(bs_reader, &pvideo, &n_video_bytes, &pts)
               != ROCDEC_SUCCESS)
            {
                std::cerr << "Failed to get picture data." << std::endl;
                return 1;
            }
            // Treat 0 bitstream size as end of stream indicator
            if(n_video_bytes == 0)
            {
                pkg_flags |= ROCDEC_PKT_ENDOFSTREAM;
            }
            n_frame_returned
                = viddec.DecodeFrame(pvideo, n_video_bytes, pkg_flags, pts, &decoded_pics);

            if(!n_frame && !viddec.GetOutputSurfaceInfo(&surf_info))
            {
                std::cerr << "Error: Failed to get Output Surface Info!" << std::endl;
                break;
            }
            for(int i = 0; i < n_frame_returned; i++)
            {
                pframe = viddec.GetFrame(&pts);
                if(dump_output_frames && mem_type != OUT_SURFACE_MEM_NOT_MAPPED)
                {
                    viddec.SaveFrameToFile(output_file_path, pframe, surf_info);
                }
                // release frame
                viddec.ReleaseFrame(pts);
            }
            auto end_time = std::chrono::high_resolution_clock::now();
            auto time_per_decode
                = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            total_dec_time += time_per_decode;
            n_frame += n_frame_returned;
            n_pic_decoded += decoded_pics;
            if(num_decoded_frames && num_decoded_frames <= static_cast<uint32_t>(n_frame))
            {
                break;
            }
        }
        while(n_video_bytes);

        n_frame += viddec.GetNumOfFlushedFrames();
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
        if(bs_reader)
        {
            rocDecDestroyBitstreamReader(bs_reader);
        }
    }
    catch(const std::exception& ex)
    {
        std::cout << ex.what() << std::endl;
        exit(1);
    }

    return 0;
}
