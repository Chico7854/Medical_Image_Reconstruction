#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <iomanip>
#include <regex>
#include <Eigen/Dense>
#include "crow.h"

using namespace std;
using namespace Eigen;

unordered_map<string, MatrixXd> H_map;

unordered_map<string, vector<string>> resultados_storage;
unordered_map<string, int> processando_count;
unordered_map<string, bool> finalizado_flags;

mutex storage_mutex;
condition_variable cv_monitor;
atomic<bool> cronometro_iniciado{false};
chrono::time_point<chrono::steady_clock> tempo_inicial;

const string METRICS_FILE = "metrics_cpp.csv";

MatrixXd load_csv(const string& path) {
    vector<vector<double>> data;
    ifstream file(path);
    string line;
    while (getline(file, line)) {
        vector<double> row;
        stringstream ss(line);
        string cell;
        while (getline(ss, cell, ',')) {
            row.push_back(stod(cell));
        }
        data.push_back(row);
    }
    if (data.empty()) return MatrixXd(0, 0);
    MatrixXd mat(data.size(), data[0].size());
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < data[0].size(); ++j) {
            mat(i, j) = data[i][j];
        }
    }
    return mat;
}

double get_cpu_percent() {
#ifdef __linux__
    static long long last_user = 0, last_nice = 0, last_sys = 0, last_idle = 0;
    ifstream file("/proc/stat");
    string label;
    long long user, nice, sys, idle;
    if (!(file >> label >> user >> nice >> sys >> idle)) return 0.0;
    long long total = (user - last_user) + (nice - last_nice) + (sys - last_sys) + (idle - last_idle);
    long long active = total - (idle - last_idle);
    last_user = user; last_nice = nice; last_sys = sys; last_idle = idle;
    return total == 0 ? 0.0 : (double)active / total * 100.0;
#else
    return 0.0; 
#endif
}

double get_used_memory_gb() {
#ifdef __linux__
    ifstream file("/proc/meminfo");
    string label;
    long long value;
    long long total = 0, free = 0, buffers = 0, cached = 0;
    while (file >> label >> value) {
        if (label == "MemTotal:") total = value;
        else if (label == "MemFree:") free = value;
        else if (label == "Buffers:") buffers = value;
        else if (label == "Cached:") cached = value;
        string unit; file >> unit;
    }
    long long used = total - free - buffers - cached;
    return (double)used / (1024.0 * 1024.0);
#else
    return 0.0;
#endif
}

void monitorar_recursos() {
    {
        unique_lock<mutex> lock(storage_mutex);
        cv_monitor.wait(lock, [] { return cronometro_iniciado.load(); });
    }
    
    tempo_inicial = chrono::steady_clock::now();
    ofstream f(METRICS_FILE);
    f << "Tempo_Decorrido_S,System_Total_CPU_Percent,System_Used_Memory_GB\n";
    f.close();

    cout << "[MONITOR] First request received! Stopwatch started. Logging to " << METRICS_FILE << "...\n" << endl;

    while (true) {
        try {
            auto agora = chrono::steady_clock::now();
            double tempo_decorrido = chrono::duration<double>(agora - tempo_inicial).count();
            
            double sys_cpu = get_cpu_percent();
            double sys_mem = get_used_memory_gb();

            ofstream fa(METRICS_FILE, ios::app);
            fa << fixed << setprecision(2) << tempo_decorrido << "," << sys_cpu << "," << sys_mem << "\n";
            fa.close();

            this_thread::sleep_for(chrono::milliseconds(500));
        } catch (...) {
            this_thread::sleep_for(chrono::seconds(1));
        }
    }
}

string get_current_datetime_str() {
    auto now = chrono::system_clock::now();
    auto in_time_t = chrono::system_clock::to_time_t(now);
    struct tm buf;
#ifdef _WIN32
    localtime_s(&buf, &in_time_t);
#else
    localtime_r(&in_time_t, &buf);
#endif
    stringstream ss;
    ss << put_time(&buf, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

tuple<VectorXd, int, double> cgne(const MatrixXd& H, const VectorXd& g, int max_iter = 100, double epsilon = 1e-4) {
    auto inicio = chrono::high_resolution_clock::now();
    double lambd = (H.transpose() * g).cwiseAbs().maxCoeff() * 0.10;
    VectorXd f = VectorXd::Zero(H.cols());
    VectorXd r = g;
    VectorXd p = H.transpose() * r;
    int iteracoes = 0;

    for (int i = 0; i < max_iter; ++i) {
        double p_dot = p.dot(p);
        double r_dot = r.dot(r);
        if ((p_dot + lambd * r_dot) == 0) break;
        
        double alpha = r_dot / (p_dot + lambd * r_dot);
        f = f + alpha * p;
        VectorXd r_novo = r - alpha * ((H * p) + lambd * r);
        
        if (r_dot == 0) break;
        double beta = r_novo.dot(r_novo) / r_dot;
        p = (H.transpose() * r_novo) + (beta * p);
        double erro = abs(r_novo.norm() - r.norm());
        r = r_novo;
        iteracoes = i + 1;
        if (erro < epsilon) break;
    }
    auto fim = chrono::high_resolution_clock::now();
    double tempo = chrono::duration<double>(fim - inicio).count();
    return {f.cwiseAbs(), iteracoes, tempo};
}

tuple<VectorXd, int, double> cgnr(const MatrixXd& H, const VectorXd& g, int max_iter = 100, double epsilon = 1e-4) {
    auto inicio = chrono::high_resolution_clock::now();
    double lambd = (H.transpose() * g).cwiseAbs().maxCoeff() * 0.10;
    VectorXd f = VectorXd::Zero(H.cols());
    VectorXd r = g - H * f;
    VectorXd z = H.transpose() * r;
    VectorXd p = z;
    int iteracoes = 0;

    for (int i = 0; i < max_iter; ++i) {
        VectorXd w = H * p;
        double z_dot = z.dot(z);
        double divisor = w.dot(w) + lambd * p.dot(p);
        if (divisor == 0) break;

        double alpha = z_dot / divisor;
        f = f + alpha * p;
        VectorXd r_novo = r - alpha * w;
        VectorXd z_novo = H.transpose() * r_novo - lambd * f;
        
        if (z_dot == 0) break;
        double beta = z_novo.dot(z_novo) / z_dot;
        p = z_novo + beta * p;
        double erro = abs(r_novo.norm() - r.norm());
        r = r_novo;
        z = z_novo;
        iteracoes = i + 1;
        if (erro < epsilon) break;
    }
    auto fim = chrono::high_resolution_clock::now();
    double tempo = chrono::duration<double>(fim - inicio).count();
    return {f.cwiseAbs(), iteracoes, tempo};
}

void processar_reconstrucao_background(string sinal_str, string h_key, string nome, string algoritmo, string client_id) {
    string inicio_str = get_current_datetime_str();
    
    auto json_arr = crow::json::load(sinal_str);
    VectorXd g(json_arr.size());
    for (size_t i = 0; i < json_arr.size(); ++i) g(i) = json_arr[i].d();

    auto it = H_map.find(h_key);
    if (it == H_map.end()) {
        lock_guard<mutex> lock(storage_mutex);
        processando_count[client_id]--;
        return;
    }
    const MatrixXd& H = it->second;

    if (H.rows() != g.size()) {
        lock_guard<mutex> lock(storage_mutex);
        processando_count[client_id]--;
        return;
    }

    int width = 0, height = 0;
    smatch match;
    regex size_regex(R"((\d+)x(\d+))");
    if (regex_search(nome, match, size_regex)) {
        width = stoi(match[1].str());
        height = stoi(match[2].str());
    } else {
        width = static_cast<int>(sqrt(H.cols()));
        height = width;
    }

    VectorXd f;
    int iteracoes;
    double tempo;

    if (algoritmo == "cgne") {
        tie(f, iteracoes, tempo) = cgne(H, g);
    } else {
        tie(f, iteracoes, tempo) = cgnr(H, g);
    }

    string fim_str = get_current_datetime_str();

    string json_output;
    json_output.reserve(200 + (width * height * 12)); 

    json_output += "{\"nome\":\"" + nome + "\"";
    json_output += ",\"iteracoes\":" + to_string(iteracoes);
    json_output += ",\"tempo\":" + to_string(tempo);
    json_output += ",\"tamanho\":\"" + to_string(width) + "x" + to_string(height) + "\"";
    json_output += ",\"inicio\":\"" + inicio_str + "\"";
    json_output += ",\"fim\":\"" + fim_str + "\"";
    json_output += ",\"algoritmo\":\"" + algoritmo + "\"";
    json_output += ",\"imagem\":[";

    char val_buf[32];
    for (int i = 0; i < height; ++i) {
        json_output += "[";
        for (int j = 0; j < width; ++j) {
            int idx = j * height + i;
            double val = (idx < f.size()) ? f(idx) : 0.0;
            int len = snprintf(val_buf, sizeof(val_buf), "%.6g", val);
            json_output.append(val_buf, len);
            if (j < width - 1) json_output += ",";
        }
        json_output += "]";
        if (i < height - 1) json_output += ",";
    }
    json_output += "]}";

    {
        lock_guard<mutex> lock(storage_mutex);
        resultados_storage[client_id].push_back(move(json_output));
        processando_count[client_id]--;
    }
}

int main() {
    cout << "Carregando Matrizes H..." << endl;
    H_map["H-1.csv"] = load_csv("H/H-1.csv");
    H_map["H-2.csv"] = load_csv("H/H-2.csv");

    thread monitor(monitorar_recursos);
    monitor.detach();

    crow::SimpleApp app;

    CROW_ROUTE(app, "/reconstruir").methods(crow::HTTPMethod::POST)([](const crow::request& req) {
        auto x = crow::json::load(req.body);
        if (!x) return crow::response(400);

        if (!cronometro_iniciado.load()) {
            cronometro_iniciado.store(true);
            cv_monitor.notify_one();
        }

        string client_id = x["client_id"].s();
        string nome = x["nome"].s();
        string h = x["h"].s();
        string algoritmo = x["algoritmo"].s();
        
        std::ostringstream os;
        os << x["sinal"];
        string sinal_str = os.str();

        {
            lock_guard<mutex> lock(storage_mutex);
            if (!processando_count.count(client_id)) {
                processando_count[client_id] = 0;
                finalizado_flags[client_id] = false;
            }
            processando_count[client_id]++;
        }

        thread worker(processar_reconstrucao_background, move(sinal_str), move(h), move(nome), move(algoritmo), move(client_id));
        worker.detach();

        return crow::response("{\"status\":\"recebido\"}");
    });

    CROW_ROUTE(app, "/finalizar/<string>").methods(crow::HTTPMethod::POST)([](string client_id) {
        lock_guard<mutex> lock(storage_mutex);
        finalizado_flags[client_id] = true;
        return crow::response("{\"status\":\"sinalizado\"}");
    });

    CROW_ROUTE(app, "/resultados/<string>").methods(crow::HTTPMethod::GET)([](string client_id) {
        string response_str;
        response_str.reserve(4096);
        
        bool terminar = false;
        
        {
            lock_guard<mutex> lock(storage_mutex);
            response_str += "{\"status\":\"sucesso\",";
            
            if (finalizado_flags.count(client_id) && finalizado_flags[client_id] && 
                processando_count.count(client_id) && processando_count[client_id] == 0 && 
                (!resultados_storage.count(client_id) || resultados_storage[client_id].empty())) {
                terminar = true;
            }
            
            response_str += "\"concluido\":" + string(terminar ? "true" : "false") + ",\"dados\":[";
            
            if (resultados_storage.count(client_id)) {
                auto& list = resultados_storage[client_id];
                for (size_t i = 0; i < list.size(); ++i) {
                    response_str += list[i];
                    if (i < list.size() - 1) response_str += ",";
                }
                
                vector<string>().swap(resultados_storage[client_id]);
            }
            response_str += "]}";
            
            if (terminar) {
                processando_count.erase(client_id);
                resultados_storage.erase(client_id);
                finalizado_flags.erase(client_id);
            }
        }

        crow::response res(move(response_str));
        res.set_header("Content-Type", "application/json");
        return res;
    });

    app.port(8000).multithreaded().run();
}