#include <iostream>
#include <vector>
#include <cmath>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <numeric>
#include <libavutil/motion_vector.h>
#include <cstdlib>   // pour std::exit, std::atof, std::atoi
#include <cstring>   // pour std::strcmp
#include <unistd.h>  // pour getopt
#include <ctime>     // pour std::time, std::localtime, strftime

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/avutil.h>
}

/*

sudo apt-get update
sudo apt-get install libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libopencv-dev

g++ -std=c++17 -o mainf mainf.cpp $(pkg-config --cflags --libs opencv4) -lavformat -lavcodec -lavutil -lswscale

avec val standard:

./mainf -v ../../vid_h264.mp4 -m 5.0 -r 20.0 -n 3 -p 1 -V 1 -d 1




La fonction generate_timestamp_folder() utilise les fonctions de la bibliothÃ¨que <ctime> pour gÃ©nÃ©rer une chaÃ®ne au format "YYYYMMDD_HHMMSS".
Dans la fonction main_function, si l'option de sortie (dump_dir) est renseignÃ©e, le code crÃ©e un dossier outYYYYMMDD_HHMMSS dans ce rÃ©pertoire de base.
Ensuite, deux sous-dossiers sont crÃ©Ã©s dans ce dossier :

frames pour les images,
motion_vectors pour les fichiers CSV contenant les vecteurs de mouvement.



La boucle while ((opt = getopt(argc, argv, "v:m:r:n:p:V:d:h")) != -1) parcourt les options disponibles.

-v permet de spÃ©cifier le chemin vers la vidÃ©o.
-m pour le seuil de magnitude.
-r pour le rayon de proximitÃ©.
-n pour le nombre minimum de voisins.
-p pour l'activation de l'aperÃ§u (preview), Ã  passer sous forme de 1/true ou 0/false.
-V pour activer ou dÃ©sactiver le mode verbeux.
-d pour le rÃ©pertoire de sortie.
-h affiche l'aide.


*/


struct MotionVector {
    int src_x, src_y;       // Position du bloc source
    int dst_x, dst_y;       // Position du bloc destination
    int motion_x, motion_y; // Composantes du vecteur de mouvement (dÃ©placement)
    int sad;                // Somme des diffÃ©rences absolues
};

// Variables FFmpeg globales
AVFormatContext* fmt_ctx = nullptr;
AVCodecContext* codec_ctx = nullptr;
AVFrame* frame = nullptr;
AVPacket pkt;
int video_stream_idx = -1;

// Fonction qui retourne une chaÃ®ne contenant la date et l'heure actuelle au format "YYYYMMDD_HHMMSS"
std::string generate_timestamp_folder() {
    std::time_t now = std::time(nullptr);
    char buf[80];
    std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", std::localtime(&now));
    return std::string(buf);
}

// Initialisation de FFmpeg et ouverture du fichier vidÃ©o
bool initialize_ffmpeg(const std::string& video_url) {
    if (avformat_open_input(&fmt_ctx, video_url.c_str(), nullptr, nullptr) != 0) {
        std::cerr << "Impossible d'ouvrir le fichier vidÃ©o : " << video_url << std::endl;
        return false;
    }

    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        std::cerr << "Impossible de trouver les informations de stream" << std::endl;
        return false;
    }

    for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {
        if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            video_stream_idx = i;
            break;
        }
    }
    if (video_stream_idx == -1) {
        std::cerr << "Aucun flux vidÃ©o trouvÃ©" << std::endl;
        return false;
    }

    AVCodecParameters* codec_params = fmt_ctx->streams[video_stream_idx]->codecpar;
    const AVCodec* codec = avcodec_find_decoder(codec_params->codec_id);
    if (!codec) {
        std::cerr << "Codec non supportÃ©" << std::endl;
        return false;
    }

    codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) {
        std::cerr << "Impossible d'allouer le codec context" << std::endl;
        return false;
    }

    if (avcodec_parameters_to_context(codec_ctx, codec_params) < 0) {
        std::cerr << "Impossible de copier les paramÃ¨tres du codec" << std::endl;
        return false;
    }

    // Activation de l'extraction des vecteurs de mouvement
    codec_ctx->flags2 |= AV_CODEC_FLAG2_EXPORT_MVS;

    if (avcodec_open2(codec_ctx, codec, nullptr) < 0) {
        std::cerr << "Impossible d'ouvrir le codec" << std::endl;
        return false;
    }

    frame = av_frame_alloc();
    if (!frame) {
        std::cerr << "Impossible d'allouer la frame" << std::endl;
        return false;
    }

    return true;
}

// Extraction des vecteurs de mouvement de la frame courante
std::vector<MotionVector> extract_motion_vectors() {
    std::vector<MotionVector> motion_vectors;
    const AVFrameSideData* sd = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
    if (!sd) {
        return {};  // Aucun vecteur de mouvement disponible
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
        mv.sad = mvs[i].motion_scale; // Utilisation de motion_scale pour le SAD (adapter si besoin)
        motion_vectors.push_back(mv);
    }

    return motion_vectors;
}

/*
 * Filtrage des vecteurs de mouvement.
 * 1. Filtre par magnitude : on conserve uniquement les vecteurs dont la magnitude 
 *    (sqrt(motion_x^2 + motion_y^2)) est supÃ©rieure ou Ã©gale Ã  magnitude_threshold.
 * 2. Filtre par proximitÃ© : on ne garde que les vecteurs qui ont au moins min_neighbors
 *    voisins dans un rayon de proximity_radius (en utilisant la position de destination).
 */
std::vector<MotionVector> filter_motion_vectors(const std::vector<MotionVector>& mvs,
                                                  float magnitude_threshold,
                                                  float proximity_radius,
                                                  int min_neighbors) {
    std::vector<MotionVector> filtered;
    std::vector<MotionVector> thresholded;

    // Filtrage par magnitude
    for (const auto& mv : mvs) {
        float mag = std::sqrt(mv.motion_x * mv.motion_x + mv.motion_y * mv.motion_y);
        if (mag >= magnitude_threshold) {
            thresholded.push_back(mv);
        }
    }

    // Filtrage par proximitÃ© (nombre de voisins)
    for (const auto& mv : thresholded) {
        int neighbor_count = 0;
        cv::Point current_point(mv.dst_x, mv.dst_y);
        for (const auto& other : thresholded) {
            cv::Point other_point(other.dst_x, other.dst_y);
            double distance = cv::norm(current_point - other_point);
            if (distance <= proximity_radius) {
                neighbor_count++;
            }
        }
        if (neighbor_count >= min_neighbors) {
            filtered.push_back(mv);
        }
    }
    return filtered;
}

/*
 * Sauvegarde des vecteurs de mouvement dans un fichier CSV.
 */
void save_motion_vectors_to_file(const std::vector<MotionVector>& mvs, const std::string& file_path) {
    std::ofstream ofs(file_path);
    if (!ofs) {
        std::cerr << "Impossible d'ouvrir le fichier " << file_path << " pour Ã©criture" << std::endl;
        return;
    }
    // Ã‰criture de l'en-tÃªte
    ofs << "src_x,src_y,dst_x,dst_y,motion_x,motion_y,sad\n";
    for (const auto& mv : mvs) {
        ofs << mv.src_x << "," << mv.src_y << ","
            << mv.dst_x << "," << mv.dst_y << ","
            << mv.motion_x << "," << mv.motion_y << ","
            << mv.sad << "\n";
    }
    ofs.close();
}

// LibÃ©ration des ressources FFmpeg
void cleanup_ffmpeg() {
    if (frame) av_frame_free(&frame);
    if (codec_ctx) avcodec_free_context(&codec_ctx);
    if (fmt_ctx) avformat_close_input(&fmt_ctx);
}

// Affichage de l'aide pour l'utilisation du programme
void usage(const char* progname) {
    std::cout << "Usage: " << progname << " [options]\n"
              << "Options:\n"
              << "  -v <video_url>            Chemin vers la vidÃ©o (dÃ©faut: vid_h264.mp4)\n"
              << "  -m <magnitude_threshold>  Seuil de magnitude minimal (dÃ©faut: 5.0)\n"
              << "  -r <proximity_radius>     Rayon de proximitÃ© en pixels (dÃ©faut: 20.0)\n"
              << "  -n <min_neighbors>        Nombre minimal de voisins (dÃ©faut: 3)\n"
              << "  -p <preview>              Afficher l'aperÃ§u (0 ou 1, dÃ©faut: 0)\n"
              << "  -V <verbose>              Mode verbeux (0 ou 1, dÃ©faut: 1)\n"
              << "  -d <dump_dir>             RÃ©pertoire de sortie de base (dÃ©faut: outputdir)\n"
              << "  -h                      Afficher ce message d'aide\n";
}

// Fonction principale de traitement
void main_function(const std::string& video_url,
                   float magnitude_threshold,
                   float proximity_radius,
                   int min_neighbors,
                   bool preview,
                   bool verbose,
                   const std::string& dump_dir, bool dump) {
    // Initialisation de FFmpeg
    if (!initialize_ffmpeg(video_url)) {
        return;
    }

    if (verbose) {
        std::cout << "Fichier vidÃ©o ouvert avec succÃ¨s: " << video_url << std::endl;
    }

    // CrÃ©ation du dossier de sortie avec timestamp
    std::string out_folder;
    if (dump) {
        std::string timestamp = generate_timestamp_folder();
        out_folder = dump_dir + "/out" + timestamp;
        std::filesystem::create_directories(out_folder + "/frames");
        std::filesystem::create_directories(out_folder + "/motion_vectors");
    }

    int step = 0;
    std::vector<double> times;

    while (av_read_frame(fmt_ctx, &pkt) >= 0) {
        if (pkt.stream_index == video_stream_idx) {
            if (verbose) {
                std::cout << "Frame : " << step << " ";
            }

            double tstart = cv::getTickCount();

            if (avcodec_send_packet(codec_ctx, &pkt) == 0) {
                while (avcodec_receive_frame(codec_ctx, frame) == 0) {
                    // Extraction et filtrage des vecteurs de mouvement
                    auto motion_vectors = extract_motion_vectors();
                    auto filtered_motion_vectors = filter_motion_vectors(motion_vectors,
                                                                         magnitude_threshold,
                                                                         proximity_radius,
                                                                         min_neighbors);
                    if (verbose) {
                        std::cout << "[Extracted: " << motion_vectors.size() << " | Filtered: " << filtered_motion_vectors.size() << "] ";
                    }

                    // Conversion de la frame en format OpenCV (BGR)
                    cv::Mat cv_frame(frame->height, frame->width, CV_8UC3);
                    uint8_t* dst_data[1] = { cv_frame.data };
                    int dst_linesize[1] = { static_cast<int>(cv_frame.step) };

                    SwsContext* sws_ctx = sws_getContext(frame->width, frame->height, codec_ctx->pix_fmt,
                                                         frame->width, frame->height, AV_PIX_FMT_BGR24,
                                                         SWS_BILINEAR, nullptr, nullptr, nullptr);
                    sws_scale(sws_ctx, frame->data, frame->linesize, 0, frame->height, dst_data, dst_linesize);
                    sws_freeContext(sws_ctx);

                    // Dessiner les vecteurs de mouvement filtrÃ©s sur la frame
                    // Ici, nous dessinons une flÃ¨che allant de (dst_x, dst_y) Ã  (dst_x + motion_x, dst_y + motion_y)
                    for (const auto& mv : filtered_motion_vectors) {
                        cv::Point start_point(mv.dst_x, mv.dst_y);
                        cv::Point end_point(mv.dst_x + mv.motion_x, mv.dst_y + mv.motion_y);
                        cv::arrowedLine(cv_frame, start_point, end_point, cv::Scalar(0, 0, 255), 1, cv::LINE_AA, 0, 0.3);
                    }
                    
                    // Sauvegarde de la frame et des vecteurs si un dossier de sortie a Ã©tÃ© dÃ©fini
                    if (dump) {
                        std::string frame_filename = out_folder + "/frames/frame-" + std::to_string(step) + ".jpg";
                        if (!cv::imwrite(frame_filename, cv_frame)) {
                            std::cerr << "Erreur lors de la sauvegarde de la frame: " << frame_filename << std::endl;
                        }
                        // Sauvegarde des vecteurs de mouvement dans un fichier CSV
                        std::string mv_filename = out_folder + "/motion_vectors/motion_vectors-" + std::to_string(step) + ".csv";
                        save_motion_vectors_to_file(filtered_motion_vectors, mv_filename);
                        if (verbose) {
                            std::cout << "[MV saved: " << mv_filename << "] ";
                        }
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
        std::cout << "\nTemps moyen (dt) : " << average_time << " secondes" << std::endl;
    }

    cleanup_ffmpeg();
    if (preview) {
        cv::destroyAllWindows();
    }
}

int main(int argc, char* argv[]) {
    // Valeurs par dÃ©faut
    std::string video_url = "vid_h264.mp4";
    float magnitude_threshold = 5.0;
    float proximity_radius = 20.0;
    int min_neighbors = 3;
    bool preview = false;
    bool verbose = true;
    bool dump = false;
    std::string dump_dir = "outputdir"; // RÃ©pertoire de base de sortie

    int opt;
    // Analyse des options avec getopt
    while ((opt = getopt(argc, argv, "v:m:r:n:p:V:d:h")) != -1) {
        switch (opt) {
            case 'v':
                video_url = optarg;
                break;
            case 'm':
                magnitude_threshold = std::atof(optarg);
                break;
            case 'r':
                proximity_radius = std::atof(optarg);
                break;
            case 'n':
                min_neighbors = std::atoi(optarg);
                break;
            case 'p':
                preview = (std::strcmp(optarg, "1") == 0 || std::strcmp(optarg, "true") == 0);
                break;
            case 'V':
                verbose = (std::strcmp(optarg, "1") == 0 || std::strcmp(optarg, "true") == 0);
                break;
            case 'd':
                dump = (std::strcmp(optarg, "1") == 0 || std::strcmp(optarg, "true") == 0);
                break;
            case 'h':
            default:
                usage(argv[0]);
                return 0;
        }
    }

    // Affichage des paramÃ¨tres si en mode verbeux
    if (verbose) {
        std::cout << "ParamÃ¨tres utilisÃ©s :" << std::endl;
        std::cout << "  VidÃ©o           : " << video_url << std::endl;
        std::cout << "  Magnitude seuil : " << magnitude_threshold << std::endl;
        std::cout << "  Rayon proximitÃ© : " << proximity_radius << std::endl;
        std::cout << "  Min voisins     : " << min_neighbors << std::endl;
        std::cout << "  AperÃ§u          : " << (preview ? "oui" : "non") << std::endl;
        std::cout << "  Dossier de base : " << dump_dir << std::endl;
    }

    main_function(video_url, magnitude_threshold, proximity_radius, min_neighbors, preview, verbose, dump_dir,dump);
    return 0;
}