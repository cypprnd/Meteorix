#include <iostream>
#include <vector>
#include <cmath>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <numeric>
#include <libavutil/motion_vector.h>
#include "../../external/CLI11.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/avutil.h>
}

#include <chrono>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

// Fonction pour obtenir un timestamp sous forme de string
std::string get_current_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm local_tm = *std::localtime(&now_c);
    
    std::ostringstream oss;
    oss << std::put_time(&local_tm, "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}



struct MotionVector {
    int src_x, src_y;    // Source block position
    int dst_x, dst_y;    // Destination block position
    int motion_x, motion_y; // Motion vector components
    int sad;             // Sum of absolute differences
};

// FFmpeg context variables
AVFormatContext* fmt_ctx = nullptr;
AVCodecContext* codec_ctx = nullptr;
AVFrame* frame = nullptr;
AVPacket pkt;
int video_stream_idx = -1;

// Initialize FFmpeg and open the video file
bool initialize_ffmpeg(const std::string& video_url) {
    // Open the video file
    if (avformat_open_input(&fmt_ctx, video_url.c_str(), nullptr, nullptr) != 0) {
        std::cerr << "Could not open video file: " << video_url << std::endl;
        return false;
    }

    // Find stream information
    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        std::cerr << "Could not find stream information" << std::endl;
        return false;
    }

    // Find the video stream
    for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx = i;
            break;
        }
    }
    if (video_stream_idx == -1) {
        std::cerr << "Could not find a video stream" << std::endl;
        return false;
    }

    // Get the codec parameters and find the decoder
    AVCodecParameters* codec_params = fmt_ctx->streams[video_stream_idx]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(codec_params->codec_id);
    if (!codec) {
        std::cerr << "Unsupported codec" << std::endl;
        return false;
    }

    // Allocate the codec context
    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        std::cerr << "Could not allocate codec context" << std::endl;
        return false;
    }

    // Copy codec parameters to the codec context
    if (avcodec_parameters_to_context(codec_ctx, codec_params) < 0) {
        std::cerr << "Could not copy codec parameters to context" << std::endl;
        return false;
    }

    // Enable motion vector extraction
    codec_ctx->flags2 |= AV_CODEC_FLAG2_EXPORT_MVS;

    // Open the codec
    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        std::cerr << "Could not open codec" << std::endl;
        return false;
    }

    // Allocate the frame
    frame = av_frame_alloc();
    if (!frame) {
        std::cerr << "Could not allocate frame" << std::endl;
        return false;
    }

    return true;
}


// Extract motion vectors from the current frame
std::vector<MotionVector> extract_motion_vectors() {
    std::vector<MotionVector> motion_vectors;

    // Check if motion vectors are available
    const AVFrameSideData* sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
    if (!sd) {
        return {};  // Return an empty vector if no motion vectors exist
    }

    AVMotionVector* mvs = (AVMotionVector*)sd->data;
    int mv_count = sd->size / sizeof(AVMotionVector);

    for (int i = 0; i < mv_count; i++) {
        MotionVector mv;
        mv.src_x = mvs[i].src_x;
        mv.src_y = mvs[i].src_y;
        mv.dst_x = mvs[i].dst_x;
        mv.dst_y = mvs[i].dst_y;
        mv.motion_x = mvs[i].motion_x;
        mv.motion_y = mvs[i].motion_y;
        mv.sad = mvs[i].motion_scale; // SAD (Sum of Absolute Differences)
        motion_vectors.push_back(mv);
    }

    return motion_vectors;
}

// Sauvegarde une frame avec les vecteurs de mouvement dessinés dessus
void save_frame_with_vectors(const cv::Mat& frame, const std::vector<MotionVector>& vectors, const std::string& filepath) {
    cv::Mat annotated_frame = frame.clone();

    for (const auto& mv : vectors) {
        cv::Point start(mv.dst_x, mv.dst_y);
        cv::Point end(mv.src_x, mv.src_y);
        cv::arrowedLine(annotated_frame, start, end, cv::Scalar(0, 0, 255), 1, cv::LINE_AA, 0.3);
    }

    cv::imwrite(filepath, annotated_frame);
}

// Cleanup FFmpeg resources
void cleanup_ffmpeg() {
    if (frame) av_frame_free(&frame);
    if (codec_ctx) avcodec_free_context(&codec_ctx);
    if (fmt_ctx) avformat_close_input(&fmt_ctx);
}

// Fonction de filtrage
std::vector<MotionVector> filterMotionVectorsByNorm(const std::vector<MotionVector>& vectors, float threshold) {
    std::vector<MotionVector> filtered;

    for (const auto& mv : vectors) {
        float norm = std::sqrt((mv.dst_x - mv.src_x)^2 + (mv.dst_y - mv.src_y)^2);
        if (norm >= threshold) {
            filtered.push_back(mv);
        }
    }

    return filtered;
}

// Main function
void main_function(const std::string& video_url, float magnitude_threshold, float proximity_radius, int min_neighbors, bool preview, bool verbose, const std::string& dump_dir) {
    if (!initialize_ffmpeg(video_url)) {
        return;
    }

    if (verbose) {
        std::cout << "Successfully opened video file" << std::endl;
    }

    if (!dump_dir.empty()) {
        std::filesystem::create_directories(dump_dir + "/frames");
        std::filesystem::create_directories(dump_dir + "/motion_vectors");
    }

    int step = 0;
    std::vector<double> times;

    while (av_read_frame(fmt_ctx, &pkt) >= 0) {
        if (pkt.stream_index == video_stream_idx) {
            if (verbose) {
                std::cout << "Frame: " << step << " ";
            }

            double tstart = cv::getTickCount();

            // Decode the video frame
            if (avcodec_send_packet(codec_ctx, &pkt) == 0) {
                while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                    // Extract motion vectors
                    auto motion_vectors = extract_motion_vectors();
                    //auto filteredVectors = filterMotionVectorsByNorm(motion_vectors, magnitude_threshold);
                
    
                    // Convert frame to OpenCV format for display
                    cv::Mat cv_frame(frame->height, frame->width, CV_8UC3);
                    uint8_t* dst_data[1] = { cv_frame.data };
                    int dst_linesize[1] = { static_cast<int>(cv_frame.step) };

                    SwsContext* sws_ctx = sws_getContext(frame->width, frame->height, codec_ctx->pix_fmt,
                                                         frame->width, frame->height, AV_PIX_FMT_BGR24,
                                                         SWS_BILINEAR, nullptr, nullptr, nullptr);
                    sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height, dst_data, dst_linesize);
                    sws_freeContext(sws_ctx);

                    // Draw motion vectors on the frame
                    for (const auto& mv : motion_vectors) {
                        cv::Point start(mv.dst_x, mv.dst_y);
                        cv::Point end(mv.src_x, mv.src_y);
                        cv::arrowedLine(cv_frame, start, end, cv::Scalar(0, 0, 255), 1, cv::LINE_AA, 0.3);
                    }

                    // Save or display the frame
                    if (!dump_dir.empty()) {
                        std::cout << get_current_timestamp();
                        std::string frame_filename = dump_dir + "/frames/frame-" + std::to_string(step) + ".jpg";
                        save_frame_with_vectors(cv_frame, filteredVectors, frame_filename);
                    }

                    if (preview) {
                        cv::imshow("Frame", cv_frame);
                        if (cv::waitKey(1) == 'q') {
                            break;
                        }
                    }

                    step++;
                }
            }

            double tend = cv::getTickCount();
            double telapsed = (tend - tstart) / cv::getTickFrequency();
            times.push_back(telapsed);
        }

        av_packet_unref(&pkt);
    }

    if (verbose) {
        double average_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        std::cout << "Average dt: " << average_time << std::endl;
    }

    cleanup_ffmpeg();
    if (preview) {
        cv::destroyAllWindows();
    }
}

int main(int argc, char* argv[]) {
    // Parse command-line arguments (use a library like CLI11 or implement manually)

    CLI::App app{"Analyse de vidéo avec détection"};

    std::string video_url = "vid_h264.mp4";
    float magnitude_threshold = 5.0;
    float proximity_radius = 20.0;
    int min_neighbors = 3;
    bool preview = true;
    bool verbose = true;
    std::string dump_dir = "outputdir";

    app.add_option("-v,--video", video_url, "URL ou chemin de la vidéo")->required();
    app.add_option("-m,--magnitude", magnitude_threshold, "Seuil de magnitude");
    app.add_option("-r,--radius", proximity_radius, "Rayon de proximité");
    app.add_option("-n,--neighbors", min_neighbors, "Nombre minimum de voisins");
    app.add_flag("--no-preview", preview, "Désactiver l'aperçu")->default_val(true);
    app.add_flag("--no-verbose", verbose, "Désactiver les logs détaillés")->default_val(true);
    app.add_option("-d,--dump-dir", dump_dir, "Répertoire de sortie");

    CLI11_PARSE(app, argc, argv);
    printf("%f",magnitude_threshold);
    main_function(video_url, magnitude_threshold, proximity_radius, min_neighbors, preview, verbose, dump_dir);
    return 0;
}