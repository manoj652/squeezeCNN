#ifndef VIDEO_KAFKA_CONSUMER
#define VIDEO_KAFKA_CONSUMER

#include <iostream>

#include <cppkafka/cppkafka.h>
#include <opencv2/opencv.hpp>

#include "base64.h"
#include "../lib/rapidjson/include/rapidjson/document.h"
#include "../lib/rapidjson/include/rapidjson/writer.h"
#include "../lib/rapidjson/include/rapidjson/stringbuffer.h"

class VideoConsumer {

    private:
        std::string _topic;
        std::string _brokers;
        std::string _groupid;
        cppkafka::Consumer *_consumer = NULL;
        cppkafka::Configuration _configuration;
    public:
        VideoConsumer(std::string brokers, std::string topic, std::string groupid);
        void setConsumer();
        cv::Mat getVideoFrame();

};

#endif