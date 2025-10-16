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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <functional>
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

#if __cplusplus >= 201703L && __has_include(<filesystem>)
    #include <filesystem>
#else
    #include <experimental/filesystem>
#endif

#include "roc_video_dec.h"
#include "video_demuxer.h"

#include "CmdParser/cmdparser.hpp"
#include "example_utils.hpp"
#include "rocdecode_utils.hpp"

class ThreadPool
{
public:
    ThreadPool(int nthreads) : shutdown_(false)
    {
        threads_.reserve(nthreads);
        for(int i = 0; i < nthreads; ++i)
        {
            threads_.emplace_back(std::bind(&ThreadPool::thread_entry, this, i));
        }
    }

    ~ThreadPool() {}

    void join_threads()
    {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            shutdown_ = true;
            cond_var_.notify_all();
        }

        for(auto& thread : threads_)
        {
            thread.join();
        }
    }

    void execute_job(std::function<void()> func)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        decode_jobs_queue_.emplace(std::move(func));
        cond_var_.notify_one();
    }

protected:
    void thread_entry(int /*i*/)
    {
        std::function<void()> execute_decode_job;

        while(true)
        {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cond_var_.wait(lock, [&] { return shutdown_ || !decode_jobs_queue_.empty(); });
                if(decode_jobs_queue_.empty())
                {
                    return;
                }

                execute_decode_job = std::move(decode_jobs_queue_.front());
                decode_jobs_queue_.pop();
            }

            execute_decode_job();
        }
    }

    std::mutex                        mutex_;
    std::condition_variable           cond_var_;
    bool                              shutdown_;
    std::queue<std::function<void()>> decode_jobs_queue_;
    std::vector<std::thread>          threads_;
};

struct DecoderInfo
{
    int                              dec_device_id;
    std::unique_ptr<RocVideoDecoder> viddec;
    std::uint32_t                    bit_depth;
    rocDecVideoCodec                 rocdec_codec_id;
    std::atomic_bool                 decoding_complete;

    DecoderInfo() : dec_device_id(0), viddec(nullptr), bit_depth(8), decoding_complete(false) {}
};

struct SeqInfo
{
    int batch_size;
    int seq_length;
    int step;
    int stride;
};

void dec_proc(RocVideoDecoder*        p_dec,
              VideoDemuxer*           demuxer,
              int*                    pn_frame,
              double*                 pn_fps,
              std::atomic_bool&       decoding_complete,
              int&                    seek_mode,
              bool&                   b_dump_output_frames,
              SeqInfo&                seq_info,
              std::string*            p_output_file_name,
              OutputSurfaceMemoryType mem_type)
{
    int                n_video_bytes = 0, n_frame_returned = 0;
    int64_t            n_frame = 0;
    uint8_t *          p_video = nullptr, *p_frame = nullptr;
    int64_t            pts            = 0;
    double             total_dec_time = 0.0;
    OutputSurfaceInfo* surf_info;
    VideoSeekContext   video_seek_ctx;
    std::vector<int>   seq_frame_start(seq_info.batch_size);
    seq_frame_start[0] = 0;
    for(int i = 1; i < seq_info.batch_size; i++)
    {
        seq_frame_start[i]
            = seq_frame_start[i - 1] + (seq_info.seq_length - 1) * seq_info.stride + seq_info.step;
    }
    auto        start_time  = std::chrono::high_resolution_clock::now();
    int         n_frame_seq = 0, num_seq = 0;
    int         next_frame_num       = 0;
    bool        seq_start            = true;
    std::string seq_output_file_name = p_output_file_name[num_seq];

    ReconfigParams            reconfig_params      = {};
    reconfig_dump_file_struct reconfig_user_struct = {};
    reconfig_params.p_fn_reconfigure_flush         = reconfigure_flush_callback;
    reconfig_user_struct.b_dump_frames_to_file     = false;
    reconfig_params.reconfig_flush_mode            = RECONFIG_FLUSH_MODE_NONE;
    reconfig_params.p_reconfig_user_struct         = &reconfig_user_struct;
    p_dec->SetReconfigParams(&reconfig_params, true);

    do
    {
        if(seek_mode && seq_start)
        {
            video_seek_ctx.seek_frame_ = seq_frame_start[num_seq];
            video_seek_ctx.seek_crit_  = SEEK_CRITERIA_FRAME_NUM;
            video_seek_ctx.seek_mode_  = SEEK_MODE_PREV_KEY_FRAME;
            demuxer->Seek(video_seek_ctx, &p_video, &n_video_bytes);
            pts       = video_seek_ctx.out_frame_pts_;
            n_frame   = static_cast<int64_t>(pts * demuxer->GetFrameRate());
            seq_start = false;
            p_dec->FlushAndReconfigure();
        }
        else
        {
            demuxer->Demux(&p_video, &n_video_bytes, &pts);
        }
        n_frame_returned = p_dec->DecodeFrame(p_video, n_video_bytes, 0, pts);
        if(b_dump_output_frames && mem_type != OUT_SURFACE_MEM_NOT_MAPPED)
        {
            if(!n_frame && !p_dec->GetOutputSurfaceInfo(&surf_info))
            {
                std::cerr << "Error: Failed to get Output Surface Info!" << std::endl;
                break;
            }
            for(int i = 0; i < n_frame_returned; i++)
            {
                if((n_frame + i) == next_frame_num)
                {
                    p_frame = p_dec->GetFrame(&pts);
                    if(n_frame_seq < seq_info.seq_length)
                    {
                        p_dec->SaveFrameToFile(seq_output_file_name, p_frame, surf_info);
                        n_frame_seq++;
                    }
                    p_dec->ReleaseFrame(pts);
                    next_frame_num += seq_info.stride;
                }
                else
                {
                    p_frame = p_dec->GetFrame(&pts);
                    p_dec->ReleaseFrame(pts);
                }
            }
        }
        n_frame += n_frame_returned;
        if(n_frame_seq >= seq_info.seq_length)
        {
            n_frame_seq = 0;
            seq_start   = true;
            num_seq++;
            if(num_seq < seq_info.batch_size)
            {
                next_frame_num       = seq_frame_start[num_seq];
                seq_output_file_name = p_output_file_name[num_seq];
            }
            p_dec->ResetSaveFrameToFile();
            n_frame_returned = p_dec->DecodeFrame(nullptr, 0, ROCDEC_PKT_ENDOFSTREAM, -1);
        }
    }
    while(n_video_bytes && num_seq < seq_info.batch_size);

    auto end_time        = std::chrono::high_resolution_clock::now();
    auto time_per_decode = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    total_dec_time               = time_per_decode;
    double average_decoding_time = total_dec_time / n_frame;
    double n_fps                 = 1000 / average_decoding_time;
    *pn_fps                      = n_fps;
    *pn_frame                    = n_frame;
    p_dec->ResetSaveFrameToFile();
    decoding_complete = true;
}

int main(int argc, char** argv)
{
    cli::Parser parser(argc, argv);
    parser.set_optional<std::string>("i", "input", "", "Input folder path (required)");
    parser.set_optional<std::string>("o", "output", "", "Output folder to dump sequences");
    parser.set_optional<int>("d", "device", 0, "GPU device ID");
    parser.set_optional<int>("t", "threads", 1, "Number of threads (1-64)");
    parser.set_optional<int>("b", "batch_size", 4, "Number of sequences to be decoded");
    parser.set_optional<int>("l", "seq_length", 4, "Number of frames in each sequence");
    parser.set_optional<int>("step", "step", 1, "Frame interval between each sequence");
    parser.set_optional<int>("stride",
                             "stride",
                             1,
                             "Distance between consecutive frames in a sequence");
    parser.set_optional<int>("seek_mode",
                             "seek_mode",
                             0,
                             "Seeking option (0: no seek, 1: seek to prev key frame)");
    parser.set_optional<int>("m",
                             "memory_type",
                             0,
                             "Output surface memory type [0: DEV_INTERNAL, 1: DEV_COPIED, 2: "
                             "HOST_COPIED, 3: NOT_MAPPED]");
    parser.set_optional<int>("disp_delay",
                             "display_delay",
                             1,
                             "Number of frames to be delayed for display");
    parser.run_and_exit_if_error();

    std::string input_folder_path = parser.get<std::string>("i");
    if(input_folder_path.empty())
    {
        std::cerr << "Error: Input folder path is required (-i option)" << std::endl;
        return 1;
    }

    std::string output_folder_path   = parser.get<std::string>("o");
    bool        b_dump_output_frames = false;
    if(!output_folder_path.empty())
    {
#if __cplusplus >= 201703L && __has_include(<filesystem>)
        if(std::filesystem::is_directory(output_folder_path))
        {
            std::filesystem::remove_all(output_folder_path);
        }
        std::filesystem::create_directory(output_folder_path);
#else
        if(std::experimental::filesystem::is_directory(output_folder_path))
        {
            std::experimental::filesystem::remove_all(output_folder_path);
        }
        std::experimental::filesystem::create_directory(output_folder_path);
#endif
        b_dump_output_frames = true;
    }

    int device_id = parser.get<int>("d");
    int n_threads = parser.get<int>("t");
    if(n_threads <= 0 || n_threads > 64)
    {
        std::cerr << "Error: Number of threads must be between 1 and 64" << std::endl;
        return 1;
    }

    SeqInfo seq_info;
    seq_info.batch_size                = parser.get<int>("b");
    seq_info.seq_length                = parser.get<int>("l");
    seq_info.step                      = parser.get<int>("step");
    seq_info.stride                    = parser.get<int>("stride");
    int                     seek_mode  = parser.get<int>("seek_mode");
    OutputSurfaceMemoryType mem_type   = static_cast<OutputSurfaceMemoryType>(parser.get<int>("m"));
    int                     disp_delay = parser.get<int>("disp_delay");

    bool                     b_extract_sei_messages = false;
    Rect*                    p_crop_rect            = nullptr;
    std::vector<std::string> input_file_names;
    int                      num_files = 0;

    try
    {
#if __cplusplus >= 201703L && __has_include(<filesystem>)
        for(const auto& entry : std::filesystem::directory_iterator(input_folder_path))
        {
#else
        for(const auto& entry :
            std::experimental::filesystem::directory_iterator(input_folder_path))
        {
#endif
            input_file_names.push_back(entry.path());
            num_files++;
        }
        n_threads = ((n_threads > num_files) ? num_files : n_threads);
        std::vector<std::string> output_seq_file_names;
        output_seq_file_names.resize(seq_info.batch_size * num_files);
        int             num_devices = 0, sd = 0;
        hipError_t      hip_status = hipSuccess;
        hipDeviceProp_t hip_dev_prop;
        std::string     gcn_arch_name;
        if(hipGetDeviceCount(&num_devices) != hipSuccess)
        {
            std::cout << "ERROR: hipGetDeviceCount failed! (" << hip_status << ")" << std::endl;
            return -1;
        }
        if(num_devices < 1)
        {
            std::cerr << "ERROR: didn't find any GPU!" << std::endl;
            return -1;
        }

        if(hipSuccess != hipGetDeviceProperties(&hip_dev_prop, device_id))
        {
            std::cerr << "ERROR: hipGetDeviceProperties for device (" << device_id << ") failed!"
                      << std::endl;
            return -1;
        }

        gcn_arch_name   = hip_dev_prop.gcnArchName;
        std::size_t pos = gcn_arch_name.find_first_of(":");
        std::string gcn_arch_name_base
            = (pos != std::string::npos) ? gcn_arch_name.substr(0, pos) : gcn_arch_name;

        if(!gcn_arch_name_base.compare("gfx90a") && num_devices > 1)
        {
            sd = 1;
        }

        std::string         device_name;
        int                 pci_bus_id, pci_domain_id, pci_device_id;
        double              total_fps = 0;
        int                 n_total   = 0;
        std::vector<double> v_fps;
        std::vector<int>    v_frame;
        v_fps.resize(num_files, 0);
        v_frame.resize(num_files, 0);
        int hip_vis_dev_count = 0;
        get_env_var("HIP_VISIBLE_DEVICES", hip_vis_dev_count);

        std::vector<std::unique_ptr<VideoDemuxer>> v_demuxer;
        std::vector<std::unique_ptr<DecoderInfo>>  v_dec_info;
        ThreadPool                                 thread_pool(n_threads);
        std::mutex                                 mutex;

        for(int i = 0; i < num_files; i++)
        {
            v_demuxer.push_back(std::make_unique<VideoDemuxer>(input_file_names[i].c_str()));
            std::size_t found_file = input_file_names[i].find_last_of('/');
            input_file_names[i]    = input_file_names[i].substr(found_file + 1);
            if(b_dump_output_frames)
            {
                std::size_t found_ext = input_file_names[i].find_last_of('.');
                std::string path
                    = output_folder_path + "/output_" + input_file_names[i].substr(0, found_ext);
                for(int n = 0; n < seq_info.batch_size; n++)
                {
                    output_seq_file_names[i * seq_info.batch_size + n]
                        = path + "_seq_" + std::to_string(n) + ".yuv";
                }
            }
        }

        for(int i = 0; i < n_threads; i++)
        {
            v_dec_info.emplace_back(std::make_unique<DecoderInfo>());
            if(!hip_vis_dev_count)
            {
                if(device_id % 2 == 0)
                {
                    v_dec_info[i]->dec_device_id = (i % 2 == 0) ? device_id : device_id + sd;
                }
                else
                {
                    v_dec_info[i]->dec_device_id = (i % 2 == 0) ? device_id - sd : device_id;
                }
            }
            else
            {
                v_dec_info[i]->dec_device_id = i % hip_vis_dev_count;
            }
            v_dec_info[i]->rocdec_codec_id = AVCodec2RocDecVideoCodec(v_demuxer[i]->GetCodecID());
            v_dec_info[i]->bit_depth       = v_demuxer[i]->GetBitDepth();
            v_dec_info[i]->viddec
                = std::make_unique<RocVideoDecoder>(v_dec_info[i]->dec_device_id,
                                                    mem_type,
                                                    v_dec_info[i]->rocdec_codec_id,
                                                    false,
                                                    p_crop_rect,
                                                    b_extract_sei_messages,
                                                    disp_delay);
            v_dec_info[i]->viddec->GetDeviceinfo(device_name,
                                                 gcn_arch_name,
                                                 pci_bus_id,
                                                 pci_domain_id,
                                                 pci_device_id);
            std::cout << "info: decoding " << input_file_names[i] << " using GPU device "
                      << v_dec_info[i]->dec_device_id << " - " << device_name << "["
                      << gcn_arch_name << "] on PCI bus " << std::setfill('0') << std::setw(2)
                      << std::right << std::hex << pci_bus_id << ":" << std::setfill('0')
                      << std::setw(2) << std::right << std::hex << pci_domain_id << "."
                      << pci_device_id << std::dec << std::endl;
        }

        for(int j = 0; j < num_files; j++)
        {
            int thread_idx = j % n_threads;
            if(j >= n_threads)
            {
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    while(!v_dec_info[thread_idx]->decoding_complete)
                    {
                        sleep(1);
                    }
                    v_dec_info[thread_idx]->decoding_complete = false;
                }
                rocDecVideoCodec codec_id = AVCodec2RocDecVideoCodec(v_demuxer[j]->GetCodecID());
                (v_dec_info[thread_idx]->viddec).release();
                v_dec_info[thread_idx]->viddec
                    = std::make_unique<RocVideoDecoder>(v_dec_info[thread_idx]->dec_device_id,
                                                        mem_type,
                                                        codec_id,
                                                        false,
                                                        p_crop_rect,
                                                        b_extract_sei_messages,
                                                        disp_delay);
                v_dec_info[thread_idx]->viddec->GetDeviceinfo(device_name,
                                                              gcn_arch_name,
                                                              pci_bus_id,
                                                              pci_domain_id,
                                                              pci_device_id);
                std::cout << "info: decoding " << input_file_names[j] << " using GPU device "
                          << v_dec_info[thread_idx]->dec_device_id << " - " << device_name << "["
                          << gcn_arch_name << "] on PCI bus " << std::setfill('0') << std::setw(2)
                          << std::right << std::hex << pci_bus_id << ":" << std::setfill('0')
                          << std::setw(2) << std::right << std::hex << pci_domain_id << "."
                          << pci_device_id << std::dec << std::endl;
            }
            if(!v_dec_info[thread_idx]->viddec->CodecSupported(
                   v_dec_info[thread_idx]->dec_device_id,
                   v_dec_info[thread_idx]->rocdec_codec_id,
                   v_dec_info[thread_idx]->bit_depth))
            {
                std::cerr << "Codec not supported on GPU, skipping this file!" << std::endl;
                continue;
            }
            thread_pool.execute_job(std::bind(dec_proc,
                                              v_dec_info[thread_idx]->viddec.get(),
                                              v_demuxer[j].get(),
                                              &v_frame[j],
                                              &v_fps[j],
                                              std::ref(v_dec_info[thread_idx]->decoding_complete),
                                              seek_mode,
                                              b_dump_output_frames,
                                              seq_info,
                                              &output_seq_file_names[j * seq_info.batch_size],
                                              mem_type));
        }

        thread_pool.join_threads();
        for(int i = 0; i < num_files; i++)
        {
            total_fps += v_fps[i] * static_cast<double>(n_threads) / static_cast<double>(num_files);
            n_total += v_frame[i];
        }
        if(!b_dump_output_frames)
        {
            std::cout << "info: Total frame decoded: " << n_total << std::endl;
            std::cout << "info: avg decoding time per frame: " << 1000 / total_fps << " ms"
                      << std::endl;
            std::cout << "info: avg FPS: " << total_fps << std::endl;
        }
        else
        {
            if(mem_type == OUT_SURFACE_MEM_NOT_MAPPED)
            {
                std::cout << "info: saving frames with -m 3 option is not supported!" << std::endl;
            }
            else
            {
                for(int i = 0; i < num_files; i++)
                {
                    for(int n = 0; n < seq_info.batch_size; n++)
                    {
                        std::cout << "info: saved frames into "
                                  << output_seq_file_names[i * seq_info.batch_size + n]
                                  << std::endl;
                    }
                }
            }
        }
    }
    catch(const std::exception& ex)
    {
        std::cout << ex.what() << std::endl;
        exit(1);
    }

    return 0;
}
