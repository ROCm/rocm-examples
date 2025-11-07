// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "CmdParser/cmdparser.hpp"

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "rocal_api.h"

using namespace std::chrono;

/// \\brief Check file extension for sequence reader validation
int check_extension(const std::string& file_name)
{
    int         position = file_name.find_last_of(".");
    std::string result   = file_name.substr(position + 1);
    if((result.compare("txt") == 0) || (result.size() == 0) || (result.compare("mp4") == 0))
        return -1;
    return 0;
}

/// \\brief Configure command-line parser for the video processing example
void configure_parser(cli::Parser& parser)
{
    parser.set_optional<std::string>("i",
                                     "input",
                                     EXAMPLE_DATA_DIR,
                                     "Input video file or image sequence directory");
    parser.set_optional<int>(
        "rc",
        "reader_case",
        2,
        "Reader case: 0=Video, 1=Video resize, 2=Sequence, 3=Sequence single shard, 4=Video resize "
        "single shard, 5=Video single shard, 6=Video single shard (alt)");
    parser.set_optional<bool>("g",
                              "gpu",
                              true,
                              "Use GPU processing (true) or CPU processing (false)");
    parser.set_optional<bool>("hd",
                              "hardware_decode",
                              false,
                              "Use hardware video decoding (true) or software (false)");
    parser.set_optional<int>("b", "batch_size", 1, "Batch size for processing");
    parser.set_optional<int>("sl", "sequence_length", 1, "Number of frames per sequence");
    parser.set_optional<int>("fs", "frame_step", 1, "Step between frames in sequence");
    parser.set_optional<int>("st", "frame_stride", 1, "Stride for frame sampling");
    parser.set_optional<bool>("c", "rgb", true, "Process RGB frames (true) or grayscale (false)");
    parser.set_optional<bool>("sv", "save_frames", false, "Save processed frames to disk");
    parser.set_optional<bool>("sh", "shuffle", false, "Shuffle frame sequences");
    parser.set_optional<int>("wh", "width", 0, "Output frame width (0 = no resize)");
    parser.set_optional<int>("ht", "height", 0, "Output frame height (0 = no resize)");
    parser.set_optional<bool>("md",
                              "enable_metadata",
                              false,
                              "Enable metadata reading for video files");
    parser.set_optional<bool>("fn", "enable_framenumbers", false, "Enable frame numbers");
    parser.set_optional<bool>("ts", "enable_timestamps", true, "Enable frame timestamps");
    parser.set_optional<bool>("sr",
                              "enable_sequence_rearrange",
                              false,
                              "Enable sequence rearrangement");
}

/// \\brief Set up the video processing pipeline
bool setup_video_pipeline(RocalContext       handle,
                          const std::string& input_path,
                          int                reader_case,
                          int                sequence_length,
                          int                frame_step,
                          int                frame_stride,
                          int                output_width,
                          int                output_height,
                          bool               use_rgb,
                          bool               hardware_decode,
                          bool               shuffle,
                          bool               enable_sequence_rearrange,
                          bool               enable_metadata,
                          bool               enable_framenumbers,
                          bool               enable_timestamps)
{
    // Determine color format based on user preference
    RocalImageColor color_format
        = (use_rgb != 0) ? RocalImageColor::ROCAL_COLOR_RGB24 : RocalImageColor::ROCAL_COLOR_U8;

    // Set up decoder mode and type
    auto decoder_mode
        = hardware_decode ? RocalDecodeDevice::ROCAL_HW_DECODE : RocalDecodeDevice::ROCAL_SW_DECODE;
    RocalDecoderType decoder_type = RocalDecoderType::ROCAL_DECODER_VIDEO_FFMPEG_SW;
    if(hardware_decode)
    {
        decoder_type = RocalDecoderType::ROCAL_DECODER_VIDEO_ROCDECODE;
    }

    bool is_output = true;
    if(enable_sequence_rearrange)
    {
        is_output = false;
    }

    size_t shard_count         = 1;
    bool   file_list_frame_num = true;

    // Validate sequence reader requirements
    if(reader_case == 3)
    {
        if(check_extension(input_path) < 0)
        {
            std::cerr << "Text file/ Video File passed as input to SEQUENCE READER" << std::endl;
            return false;
        }
        if(enable_metadata)
        {
            std::cout << "METADATA cannot be enabled for SEQUENCE READER" << std::endl;
            enable_metadata = false;
        }
        if(enable_framenumbers)
            enable_framenumbers = false;
        if(enable_timestamps)
            enable_timestamps = false;
    }
    else if(enable_metadata)
    {
        std::cout << "META DATA READER" << std::endl;
        rocalCreateVideoLabelReader(handle,
                                    input_path.c_str(),
                                    sequence_length,
                                    frame_step,
                                    frame_stride,
                                    file_list_frame_num);
    }

    // Create video tensor based on reader case
    RocalTensor video_tensor;
    switch(reader_case)
    {
        default:
            {
                std::cout << "VIDEO READER" << std::endl;
                video_tensor = rocalVideoFileSource(handle,
                                                    input_path.c_str(),
                                                    color_format,
                                                    decoder_mode,
                                                    shard_count,
                                                    sequence_length,
                                                    shuffle,
                                                    is_output,
                                                    false,
                                                    decoder_type,
                                                    frame_step,
                                                    frame_stride,
                                                    file_list_frame_num);
                break;
            }
        case 1:
            {
                std::cout << "VIDEO READER RESIZE" << std::endl;
                if(output_width == 0 || output_height == 0)
                {
                    std::cerr << "Resize width and height are passed as NULL values" << std::endl;
                    return false;
                }
                video_tensor = rocalVideoFileResize(handle,
                                                    input_path.c_str(),
                                                    color_format,
                                                    decoder_mode,
                                                    shard_count,
                                                    sequence_length,
                                                    output_width,
                                                    output_height,
                                                    shuffle,
                                                    is_output,
                                                    false,
                                                    decoder_type,
                                                    frame_step,
                                                    frame_stride,
                                                    file_list_frame_num);
                break;
            }
        case 2:
            {
                std::cout << "SEQUENCE READER" << std::endl;
                video_tensor = rocalSequenceReader(handle,
                                                   input_path.c_str(),
                                                   color_format,
                                                   shard_count,
                                                   sequence_length,
                                                   is_output,
                                                   shuffle,
                                                   false,
                                                   frame_step,
                                                   frame_stride);
                break;
            }
        case 3:
            {
                std::cout << "SEQUENCE READER - Single Shard" << std::endl;
                video_tensor = rocalSequenceReaderSingleShard(handle,
                                                              input_path.c_str(),
                                                              color_format,
                                                              0,
                                                              2,
                                                              sequence_length,
                                                              is_output,
                                                              shuffle,
                                                              false,
                                                              frame_step,
                                                              frame_stride);
                break;
            }
        case 4:
            {
                std::cout << "VIDEO READER RESIZE - SINGLE SHARD" << std::endl;
                if(output_width == 0 || output_height == 0)
                {
                    std::cerr << "Resize width and height are passed as NULL values" << std::endl;
                    return false;
                }
                video_tensor = rocalVideoFileResizeSingleShard(handle,
                                                               input_path.c_str(),
                                                               color_format,
                                                               decoder_mode,
                                                               0,
                                                               1,
                                                               sequence_length,
                                                               output_width,
                                                               output_height,
                                                               shuffle,
                                                               is_output,
                                                               false,
                                                               decoder_type,
                                                               frame_step,
                                                               frame_stride,
                                                               file_list_frame_num);
                break;
            }
        case 5:
            {
                std::cout << "VIDEO READER - SINGLE SHARD" << std::endl;
                video_tensor = rocalVideoFileSourceSingleShard(handle,
                                                               input_path.c_str(),
                                                               color_format,
                                                               decoder_mode,
                                                               0,
                                                               2,
                                                               sequence_length,
                                                               shuffle,
                                                               is_output,
                                                               false,
                                                               decoder_type,
                                                               frame_step,
                                                               frame_stride,
                                                               file_list_frame_num);
                break;
            }
        case 6:
            {
                std::cout << "VIDEO READER - SINGLE SHARD (ALT)" << std::endl;
                video_tensor = rocalVideoFileSourceSingleShard(handle,
                                                               input_path.c_str(),
                                                               color_format,
                                                               decoder_mode,
                                                               0,
                                                               2,
                                                               sequence_length,
                                                               shuffle,
                                                               is_output,
                                                               false,
                                                               decoder_type,
                                                               frame_step,
                                                               frame_stride,
                                                               file_list_frame_num);
                break;
            }
    }

    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cerr << "Error while adding the augmentation nodes" << std::endl;
        auto err_msg = rocalGetErrorMessage(handle);
        std::cout << err_msg << std::endl;
        return false;
    }

    // Apply sequence rearrangement if enabled
    if(enable_sequence_rearrange)
    {
        std::cout << "ENABLE SEQUENCE REARRANGE" << std::endl;
        std::vector<unsigned> new_order = {0, 0, 1, 1, 0};
        video_tensor = rocalSequenceRearrange(handle, video_tensor, new_order, true);
    }

    // Create dynamic color temperature adjustment parameter (from reference.cpp)
    RocalIntParam color_temp_adj = rocalCreateIntParameter(0);

    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cerr << "Error while adding the augmentation nodes" << std::endl;
        auto err_msg = rocalGetErrorMessage(handle);
        std::cout << err_msg << std::endl;
        return false;
    }

    return true;
}

/// \\brief Process video frames and save results
void process_video_frames(RocalContext handle,
                          int          sequence_length,
                          bool         save_frames,
                          bool         enable_metadata,
                          bool         enable_framenumbers,
                          bool         enable_timestamps,
                          bool         enable_sequence_rearrange,
                          int          batch_size)
{
    // Get output dimensions
    int h = rocalGetAugmentationBranchCount(handle) * rocalGetOutputHeight(handle) * batch_size;
    int w = rocalGetOutputWidth(handle);
    int p = ((rocalGetOutputColorFormat(handle) == RocalImageColor::ROCAL_COLOR_RGB24) ? 3 : 1);
    int single_image_height = h / batch_size;

    unsigned output_frames_per_sequence = sequence_length;
    if(enable_sequence_rearrange)
    {
        output_frames_per_sequence = 5; // new_order.size()
    }

    std::cout << "output width " << w << " output height " << h << " color planes " << p
              << std::endl;
    auto    cv_color_format = ((p == 3) ? CV_8UC3 : CV_8UC1);
    cv::Mat mat_input(h, w, cv_color_format);
    cv::Mat mat_color, mat_output;

    // Create dynamic color temperature adjustment parameter
    RocalIntParam color_temp_adj       = rocalCreateIntParameter(0);
    int           color_temp_increment = 1;

    // Process video sequences
    auto start_time    = std::chrono::high_resolution_clock::now();
    int  frame_counter = 0;
    int  counter       = 0;
    int  batch_counter = 0;

    while(!rocalIsEmpty(handle))
    {
        batch_counter++;
        if(rocalRun(handle) != 0)
        {
            std::cout << "rocalRun Failed with runtime error" << std::endl;
            rocalRelease(handle);
            return;
        }

        // Dynamic color temperature adjustment
        if(rocalGetIntValue(color_temp_adj) <= -99 || rocalGetIntValue(color_temp_adj) >= 99)
        {
            color_temp_increment *= -1;
        }
        rocalUpdateIntParameter(rocalGetIntValue(color_temp_adj) + color_temp_increment,
                                color_temp_adj);

        rocalCopyToOutput(handle, mat_input.data, h * w * p);
        counter += batch_size;

        // Save frames if requested
        if(save_frames)
        {
            for(unsigned b = 0; b < batch_size; b++)
            {
                // Save individual frames with better naming
                for(unsigned i = 0; i < output_frames_per_sequence; i++)
                {
                    std::string save_image_path = "video_batch_" + std::to_string(batch_counter)
                                                  + "_seq_" + std::to_string(b) + "_frame_"
                                                  + std::to_string(i) + ".png";

                    mat_output
                        = mat_input(cv::Rect(0,
                                             ((b * single_image_height * output_frames_per_sequence)
                                              + (i * single_image_height)),
                                             w,
                                             single_image_height));

                    if(p == 3)
                    {
                        cv::cvtColor(mat_output, mat_color, cv::COLOR_RGB2BGR);
                        cv::imwrite(save_image_path, mat_color);
                    }
                    else
                    {
                        cv::imwrite(save_image_path, mat_output);
                    }

                    frame_counter++;
                }

                // Optional: Create video file per sequence
                // This saves a video file for each sequence in the batch
                std::string video_save_path = "video_batch_" + std::to_string(batch_counter)
                                              + "_seq_" + std::to_string(b) + "_output.avi";

                int      frame_width  = static_cast<int>(w);
                int      frame_height = static_cast<int>(single_image_height);
                cv::Size frame_size(frame_width, frame_height);
                int      frames_per_second = 10;

                // Create and initialize the VideoWriter object
                cv::VideoWriter video_writer(video_save_path,
                                             cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                             frames_per_second,
                                             frame_size,
                                             p == 3);

                if(video_writer.isOpened())
                {
                    // Write all frames from this sequence to the video file
                    for(unsigned i = 0; i < output_frames_per_sequence; i++)
                    {
                        mat_output = mat_input(
                            cv::Rect(0,
                                     ((b * single_image_height * output_frames_per_sequence)
                                      + (i * single_image_height)),
                                     w,
                                     single_image_height));

                        if(p == 3)
                        {
                            cv::cvtColor(mat_output, mat_color, cv::COLOR_RGB2BGR);
                            video_writer.write(mat_color);
                        }
                        else
                        {
                            video_writer.write(mat_output);
                        }
                    }
                    video_writer.release();
                }
            }
        }

        // Get metadata if enabled
        if(enable_metadata)
        {
            std::vector<int>  image_name_length(batch_size);
            RocalTensorList   labels   = rocalGetImageLabels(handle);
            int               img_size = rocalGetImageNameLen(handle, image_name_length.data());
            std::vector<char> img_name(img_size);
            rocalGetImageName(handle, img_name.data());

            std::cout << "Image names: " << img_name.data() << std::endl;
            std::cout << "Label id: ";
            int* label_id = reinterpret_cast<int*>(labels->at(0)->buffer());
            for(unsigned i = 0; i < batch_size; i++)
            {
                std::cout << label_id[i] << "\t";
            }
            std::cout << std::endl;
        }

        // Get frame numbers and timestamps if enabled
        if(enable_framenumbers || enable_timestamps)
        {
            std::vector<unsigned int> start_frame_num(batch_size);
            std::vector<float>        frame_timestamps(batch_size * sequence_length);
            rocalGetSequenceStartFrameNumber(handle, start_frame_num.data());
            if(enable_timestamps)
            {
                rocalGetSequenceFrameTimestamps(handle, frame_timestamps.data());
            }
            for(unsigned i = 0; i < batch_size; i++)
            {
                if(enable_framenumbers)
                    std::cout << "Frame number : " << start_frame_num[i] << std::endl;
                if(enable_timestamps)
                    for(unsigned j = 0; j < sequence_length; j++)
                        std::cout << "T" << j << " : "
                                  << frame_timestamps[(i * sequence_length) + j] << "\t";
                std::cout << std::endl;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();

    // Print detailed timing information
    auto rocal_timing = rocalGetTimingInfo(handle);
    std::cout << "Load     time " << rocal_timing.load_time << std::endl;
    std::cout << "Decode   time " << rocal_timing.decode_time << std::endl;
    std::cout << "Process  time " << rocal_timing.process_time << std::endl;
    std::cout << "Transfer time " << rocal_timing.transfer_time << std::endl;
    std::cout << "Processed " << counter << " images/frames" << std::endl
              << "Total Elapsed Time " << dur / 1000000 << " sec " << dur % 1000000 << " us "
              << std::endl;

    if(save_frames)
    {
        std::cout << "Video frames saved to current directory" << std::endl;
    }
}

int main(int argc, const char** argv)
{
    // Parse command-line arguments
    cli::Parser parser(argc, argv);
    configure_parser(parser);
    parser.run_and_exit_if_error();

    // Extract parameters
    const std::string input_path                = parser.get<std::string>("i");
    const int         reader_case               = parser.get<int>("rc");
    const bool        use_gpu                   = parser.get<bool>("g");
    const bool        hardware_decode           = parser.get<bool>("hd");
    const int         batch_size                = parser.get<int>("b");
    const int         sequence_length           = parser.get<int>("sl");
    const int         frame_step                = parser.get<int>("fs");
    const int         frame_stride              = parser.get<int>("st");
    const int         output_width              = parser.get<int>("wh");
    const int         output_height             = parser.get<int>("ht");
    const bool        use_rgb                   = parser.get<bool>("c");
    const bool        save_frames               = parser.get<bool>("sv");
    const bool        shuffle                   = parser.get<bool>("sh");
    const bool        enable_metadata           = parser.get<bool>("md");
    const bool        enable_framenumbers       = parser.get<bool>("fn");
    const bool        enable_timestamps         = parser.get<bool>("ts");
    const bool        enable_sequence_rearrange = parser.get<bool>("sr");

    // Print basic configuration
    std::cout << "Batch size : " << batch_size << std::endl;
    std::cout << "Sequence length : " << sequence_length << std::endl;
    std::cout << "Frame step : " << frame_step << std::endl;
    std::cout << "Frame stride : " << frame_stride << std::endl;
    if(reader_case == 1 || reader_case == 4)
    {
        std::cout << "Resize Width : " << output_width << std::endl;
        std::cout << "Resize height : " << output_height << std::endl;
    }

    // Create rocAL context
    RocalProcessMode process_mode
        = use_gpu ? RocalProcessMode::ROCAL_PROCESS_GPU : RocalProcessMode::ROCAL_PROCESS_CPU;
    auto handle = rocalCreate(batch_size, process_mode, 0, 1);

    if(rocalGetStatus(handle) != ROCAL_OK)
    {
        std::cout << "Could not create the Rocal contex" << std::endl;
        return -1;
    }

    // Set up the video processing pipeline
    if(!setup_video_pipeline(handle,
                             input_path,
                             reader_case,
                             sequence_length,
                             frame_step,
                             frame_stride,
                             output_width,
                             output_height,
                             use_rgb,
                             hardware_decode,
                             shuffle,
                             enable_sequence_rearrange,
                             enable_metadata,
                             enable_framenumbers,
                             enable_timestamps))
    {
        rocalRelease(handle);
        return -1;
    }

    // Verify and build the augmentation graph
    if(rocalVerify(handle) != ROCAL_OK)
    {
        std::cerr << "Could not verify the augmentation graph" << std::endl;
        rocalRelease(handle);
        return -1;
    }

    std::cout << "Remaining images " << rocalGetRemainingImages(handle) << std::endl;
    std::cout << "Augmented copies count " << rocalGetAugmentationBranchCount(handle) << std::endl;

    // Process video frames
    process_video_frames(handle,
                         sequence_length,
                         save_frames,
                         enable_metadata,
                         enable_framenumbers,
                         enable_timestamps,
                         enable_sequence_rearrange,
                         batch_size);

    // Clean up
    rocalResetLoaders(handle);
    rocalRelease(handle);

    std::cout << "Video processing example completed successfully!" << std::endl;
    return 0;
}
