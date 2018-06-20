#include "network_utils.hpp"

using namespace std;
using namespace rapidjson;

int cpt = 0;

NetworkUtils::NetworkUtils(Device device) {
    RestClient::init();
    _device = device;
    
}

int NetworkUtils::getAuthToken(string url,std::string& token) {
    string jsonDevice = "{ \"data\":{\"deviceName\":\""+_device.deviceName+"\",\"deviceMac\":\""+_device.deviceMac+"\"} }";
    _conn = new RestClient::Connection(url);
    RestClient::HeaderFields headers;
    headers["Content-Type"] = "application/json";
    _conn->SetHeaders(headers);
    RestClient::Response r = _conn->post("/device/authenticate",jsonDevice);
    Document document;
    document.Parse(r.body.c_str());
    #ifdef EBUG
    cout << r.body << endl;
    #endif
    if(!document.HasMember("data")) {
        cout << "Parsing json problem" << endl;
        return 500;
    }
    
    const Value& data = document["data"];

        
    token = data["token"].GetString();
    _token = token;
    RestClient::disable();
    return r.code;
}

int NetworkUtils::checkEmployee(string url, Employee& employee) {
    string jsonEmployee = "{ \"data\":{\"firstName\":\""+employee.firstName+"\",\"lastName\":\""+employee.lastName+"\"} }";
    _conn = new RestClient::Connection(url);
    RestClient::HeaderFields headers;
    headers["Content-Type"] = "application/json";
    headers["Authorization"] = "Bearer "+_token;
    _conn->SetHeaders(headers);

    RestClient::Response r = _conn->post("/recognition/face",jsonEmployee);
    if(r.code == 200) employee.auth = true;
    if(r.code == 401 && cpt < 5) {
        string token;
        getAuthToken(url,token);
        checkEmployee(url,employee);
        cpt++;   
    }
    cpt = 0;
    return r.code;
}

