#include <iostream>
#include <fstream>
#include <vector>
#include <string>

int main() {
    std::vector<std::string> cases = {"06", "09", "10", "12"};
    int world_sizes[] = {8, 24};

    for (int ws : world_sizes) {
        std::cout << "=== World Size P = " << ws << " ===" << std::endl;
        for (const auto &c : cases) {
            std::string path = "cases/hw2-1/" + c + ".txt";
            std::ifstream fin(path);
            if (!fin) {
                std::cerr << "Cannot open " << path << std::endl;
                continue;
            }
            int scale, edgefactor, seed, scramble, nsample, S;
            fin >> scale >> edgefactor >> seed >> scramble >> nsample >> S;
            bool use_coop = (ws >= 4 && ws > S && scale >= 20);
            int team_count = use_coop ? std::min<int>(ws, S) : 1;
            int team_size = use_coop ? (ws / team_count) : 1;
            std::cout << "Case " << c << " (scale=" << scale << ", S=" << S << "): use_coop="
                      << (use_coop ? "TRUE" : "FALSE")
                      << " (teams=" << team_count << ", team_size=" << team_size << ")"
                      << std::endl;
        }
    }
    return 0;
}
