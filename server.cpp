/*
 * CGNR Image Reconstruction Server - C++ Translation
 *
 * Dependencies:
 *   - Eigen3         (linear algebra, replaces numpy)
 *   - cpp-httplib    (HTTP server, replaces FastAPI)
 *   - nlohmann/json  (JSON handling, replaces pydantic)
 *
 * Compile:
 *   g++ -O2 -std=c++17 -I/usr/include/eigen3 -I. -lpthread -lssl -lcrypto server.cpp -o cgnr_server
 *
 * Usage:
 *   ./cgnr_server
 *   # Listens on http://localhost:8000
 *   # POST /reconstruir with JSON body:
 *   # { "sinal": [...], "h": "H-1.csv", "nome": "teste" }
 */

#define CPPHTTPLIB_OPENSSL_SUPPORT 0
#include "httplib.h"
#include <nlohmann/json.hpp>
#include <Eigen/Dense>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <chrono>
#include <ctime>
#include <cmath>
#include <stdexcept>

using json = nlohmann::json;
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// ─── CSV loader (replaces np.loadtxt) ────────────────────────────────────────

Matrix load_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ','))
            row.push_back(std::stod(cell));
        rows.push_back(row);
    }

    if (rows.empty()) throw std::runtime_error("Empty CSV: " + path);

    Eigen::Index nrows = rows.size();
    Eigen::Index ncols = rows[0].size();
    Matrix mat(nrows, ncols);
    for (Eigen::Index i = 0; i < nrows; ++i)
        for (Eigen::Index j = 0; j < ncols; ++j)
            mat(i, j) = rows[i][j];
    return mat;
}

// ─── CGNR algorithm ──────────────────────────────────────────────────────────

struct CGNRResult {
    Vector f;
    int    iteracoes;
    double tempo;
};

CGNRResult cgnr(const Matrix& H, const Vector& g,
                int max_iter = 10, double epsilon = 1e-4) {
    auto inicio = std::chrono::high_resolution_clock::now();

    Vector f = Vector::Zero(H.cols());
    Vector r = g - H * f;
    Vector z = H.transpose() * r;
    Vector p = z;

    int iteracoes = 0;
    for (int i = 0; i < max_iter; ++i) {
        Vector w      = H * p;
        double zz     = z.dot(z);
        double ww     = w.dot(w);
        double alpha  = zz / ww;

        f             = f + alpha * p;
        Vector r_novo = r - alpha * w;
        Vector z_novo = H.transpose() * r_novo;

        double beta   = z_novo.dot(z_novo) / zz;
        p             = z_novo + beta * p;

        double rr     = r.dot(r);
        double erro   = std::abs(r_novo.dot(r_novo) - rr) / rr;

        r         = r_novo;
        z         = z_novo;
        iteracoes = i + 1;

        if (erro < epsilon) break;
    }

    auto fim     = std::chrono::high_resolution_clock::now();
    double tempo = std::chrono::duration<double>(fim - inicio).count();
    tempo        = std::round(tempo * 10000.0) / 10000.0;

    return {f, iteracoes, tempo};
}

// ─── Timestamp helper ─────────────────────────────────────────────────────────

std::string now_str() {
    auto now      = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    char buf[20];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return std::string(buf);
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    std::unordered_map<std::string, Matrix> H_map;
    try {
        H_map["H-1.csv"] = load_csv("H/H-1.csv");
        H_map["H-2.csv"] = load_csv("H/H-2.csv");
        std::cout << "H matrices loaded.\n";
    } catch (const std::exception& e) {
        std::cerr << "Warning loading matrices: " << e.what() << "\n";
    }

    httplib::Server app;

    app.Post("/reconstruir", [&](const httplib::Request& req,
                                  httplib::Response&      res) {
        try {
            json body       = json::parse(req.body);
            std::string nome  = body["nome"];
            std::string h_key = body["h"];
            std::vector<double> sinal = body["sinal"].get<std::vector<double>>();

            if (H_map.find(h_key) == H_map.end())
                throw std::runtime_error("Unknown H matrix: " + h_key);

            std::string inicio_str = now_str();

            const Matrix& H = H_map.at(h_key);
            Vector g = Eigen::Map<Vector>(sinal.data(), sinal.size());
            int tamanho = static_cast<int>(std::round(std::sqrt(H.cols())));

            auto [f, iteracoes, tempo] = cgnr(H, g);

            // Reshape f into tamanho×tamanho (column-major) then transpose
            Eigen::Map<Matrix> F_col(f.data(), tamanho, tamanho);
            Matrix imagem_mat = F_col.transpose();

            // Serialize with 90° clockwise rotation:
            // clockwise rotation: new[r][c] = original[tamanho-1-c][r]
            json imagem = json::array();
            for (int r = 0; r < tamanho; ++r) {
                json row = json::array();
                for (int c = 0; c < tamanho; ++c)
                    row.push_back(imagem_mat(c, r));
                imagem.push_back(row);
            }

            std::string fim_str = now_str();

            json resultado = {
                {"nome",      nome},
                {"imagem",    imagem},
                {"iteracoes", iteracoes},
                {"tempo",     tempo},
                {"tamanho",   std::to_string(tamanho) + "x" + std::to_string(tamanho)},
                {"inicio",    inicio_str},
                {"fim",       fim_str}
            };

            std::cout << "[" << fim_str << "] " << nome
                      << " — " << iteracoes << " iterações — "
                      << tempo << "s\n";

            res.set_content(resultado.dump(), "application/json");

        } catch (const std::exception& e) {
            json err = {{"error", e.what()}};
            res.status = 400;
            res.set_content(err.dump(), "application/json");
        }
    });

    std::cout << "Server running on http://0.0.0.0:8000\n";
    app.listen("0.0.0.0", 8000);
    return 0;
}
