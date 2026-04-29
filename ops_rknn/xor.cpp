#include "rknn_api.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <sys/stat.h>
#include <cstdlib>
#include <chrono>
#include <cstdint>
#include <cstring>

static bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

static std::string selectModelPath(const std::string& operation, int size) {
    std::stringstream specific_model_path_ss;
    specific_model_path_ss << "models/" << operation << "_int32_1x" << size << ".rknn";
    std::string specific_model_path = specific_model_path_ss.str();
    if (fileExists(specific_model_path)) return specific_model_path;
    std::stringstream generic_model_path_ss;
    generic_model_path_ss << "models/" << operation << "_int32_1x1.rknn";
    std::string generic_model_path = generic_model_path_ss.str();
    std::cout << "Using generic model: " << generic_model_path << std::endl;
    return generic_model_path;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " xor 1x<size>" << std::endl;
        return -1;
    }
    std::string operation = argv[1];
    if (operation != "xor") {
        std::cerr << "Error: Operation must be 'xor'" << std::endl;
        return -1;
    }
    std::string input_size_str = argv[2];
    if (input_size_str.substr(0, 2) != "1x") {
        std::cerr << "Error: Input size must be in format 1xN" << std::endl;
        return -1;
    }
    int size = std::atoi(input_size_str.substr(2).c_str());
    if (size <= 0) {
        std::cerr << "Error: Size must be a positive integer" << std::endl;
        return -1;
    }
    std::string model_path = selectModelPath(operation, size);
    FILE* fp = fopen(model_path.c_str(), "rb");
    if (!fp) {
        std::cerr << "Failed to open model file: " << model_path << std::endl;
        return -1;
    }
    struct stat st;
    stat(model_path.c_str(), &st);
    size_t model_size = st.st_size;
    void* model_data = malloc(model_size);
    if (!model_data) {
        std::cerr << "Failed to allocate memory for model" << std::endl;
        fclose(fp);
        return -1;
    }
    fread(model_data, 1, model_size, fp);
    fclose(fp);
    rknn_context ctx;
    int ret = rknn_init(&ctx, model_data, model_size, 0, NULL);
    free(model_data);
    if (ret < 0) {
        std::cerr << "rknn_init failed! ret=" << ret << std::endl;
        return -1;
    }
    int total = size * size;
    std::vector<int32_t> input_x(total);
    std::vector<int32_t> input_y(total);
    for (int i = 0; i < total; ++i) {
        input_x[i] = i + 1;
        input_y[i] = total - i;
    }
    rknn_input in[2] {};
    in[0].index = 0;
    in[0].type = RKNN_TENSOR_INT32;
    in[0].fmt = RKNN_TENSOR_NCHW;
    in[0].buf = input_x.data();
    in[0].size = input_x.size() * sizeof(int32_t);
    in[1].index = 1;
    in[1].type = RKNN_TENSOR_INT32;
    in[1].fmt = RKNN_TENSOR_NCHW;
    in[1].buf = input_y.data();
    in[1].size = input_y.size() * sizeof(int32_t);
    ret = rknn_inputs_set(ctx, 2, in);
    if (ret < 0) {
        std::cerr << "rknn_inputs_set failed: " << ret << std::endl;
        rknn_destroy(ctx);
        return -1;
    }
    ret = rknn_run(ctx, nullptr);
    if (ret < 0) {
        std::cerr << "rknn_run failed: " << ret << std::endl;
        rknn_destroy(ctx);
        return -1;
    }
    rknn_output out {};
    out.want_float = 0;
    out.index = 0;
    ret = rknn_outputs_get(ctx, 1, &out, nullptr);
    if (ret < 0) {
        std::cerr << "rknn_outputs_get failed: " << ret << std::endl;
        rknn_destroy(ctx);
        return -1;
    }
    int32_t* result = static_cast<int32_t*>(out.buf);
    std::cout << "Input x: ";
    for (auto v : input_x) std::cout << v << " ";
    std::cout << std::endl;
    std::cout << "Input y: ";
    for (auto v : input_y) std::cout << v << " ";
    std::cout << std::endl;
    std::cout << "Output: ";
    for (int i = 0; i < total; ++i) std::cout << result[i] << " ";
    std::cout << std::endl;
    rknn_outputs_release(ctx, 1, &out);
    rknn_destroy(ctx);
    return 0;
}
