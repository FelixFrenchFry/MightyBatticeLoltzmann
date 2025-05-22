#include <filesystem>
#include <fstream>
#include <spdlog/spdlog.h>



void ExportScalarField(
    const std::vector<float>& buffer,
    const std::string& fileName,
    const int N_X, const int N_Y)
{
    SPDLOG_INFO("Current working output directory: {}",
        std::filesystem::current_path().string());

    std::streamsize size = buffer.size() * sizeof(float);

    // export data in .csv format
    std::ofstream file_csv(fileName + ".csv");
    if (!file_csv)
    {
        SPDLOG_ERROR("Could not open output file: {}.csv", fileName);
        return;
    }

    for (int y = 0; y < N_Y; y++)
    {
        for (int x = 0; x < N_X; x++)
        {
            int idx = y * N_X + x;
            file_csv << buffer[idx];

            if (x < N_X - 1) { file_csv << ","; }
        }

        file_csv << "\n";
    }

    file_csv.close();

    // export data in .bin format
    std::ofstream file_bin(fileName + ".bin", std::ios::binary);
    if (!file_bin)
    {
        SPDLOG_ERROR("Could not open output file: {}.bin", fileName);
        return;
    }

    file_bin.write(reinterpret_cast<const char*>(buffer.data()), size);
    file_bin.close();
}
