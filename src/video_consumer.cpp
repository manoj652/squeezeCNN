#include "video_consumer.hpp"

using namespace std;
using namespace cppkafka;
using namespace rapidjson;

VideoConsumer::VideoConsumer(std::string brokers, std::string topic, std::string groupid) {
    _brokers = brokers;
    _topic = topic;
    _groupid = groupid;
    _configuration = {
        {"metadata.broker.list", brokers},
        {"max.partition.fetch.bytes","2097152"},
        {"group.id",groupid}
    };
    
}

void VideoConsumer::setConsumer() {
    _consumer = (new Consumer(_configuration));
    _consumer->subscribe({_topic});
}

cv::Mat VideoConsumer::getVideoFrame() {
    cout << "VideoConsumer" << endl;
    Message msg = _consumer->poll();
    if(!msg) {
        cerr << "No message received" << endl;
        return cv::Mat();
    }
    if(msg.get_error()) {
        if(!msg.is_eof()) {
            cerr << "[+] Received error notification: " << msg.get_error() << endl;
        }
        return cv::Mat();
    }
    
    Document document;
    string jsonPayload = "";
    for(auto i=msg.get_payload().begin(); i != msg.get_payload().end();i++) {
        jsonPayload += *i;
    }
    
    _consumer->commit(msg);
    
    cout << jsonPayload << endl;
        

    int rows = document["rows"].GetInt();
    int cols = document["cols"].GetInt();
    int type = document["type"].GetInt();
    string data = document["data"].GetString();
    
    std::vector<BYTE> decodedBytes = base64_decode(data);
    string decoded_data(decoded_data.begin(),decoded_data.end());
    cv::Mat m(rows,cols,type,(void *)decoded_data.data());
    return m;
    //return cv::Mat();
    
}