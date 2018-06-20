#ifndef NETWORK_SQUEEZE
#define NETWORK_SQUEEZE

#include <iostream>
#include <restclient-cpp/connection.h>
#include <restclient-cpp/restclient.h>

#include "../lib/rapidjson/include/rapidjson/document.h"
#include "../lib/rapidjson/include/rapidjson/writer.h"
#include "../lib/rapidjson/include/rapidjson/stringbuffer.h"
#define URL "http://localhost:8080"

typedef struct {
    std::string deviceMac;
    std::string deviceName;
} Device;

typedef struct {
    std::string lastName;
    std::string firstName;
    bool auth;
} Employee;

class NetworkUtils {
    private:
    RestClient::Connection* _conn;
    std::string _token;
    Device _device;
    public:
    NetworkUtils(Device);

    int getAuthToken(std::string,std::string&);
    int checkEmployee(std::string,Employee&);
};

#endif