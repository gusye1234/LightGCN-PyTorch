/*
<%
cfg['compiler_args'] = ['-std=c++11', '-undefined dynamic_lookup']
%>
<%
setup_pybind11(cfg)
%>
*/
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <random>
#include <algorithm>
#include <time.h>

typedef unsigned int ui;

using namespace std;
namespace py = pybind11;

int randint_(int end)
{
    return rand() % end;
}

py::array_t<int> sample_negative(int user_num, int item_num, int train_num, std::vector<std::vector<int>> allPos, int neg_num)
{
    int perUserNum = (train_num / user_num);
    int row = neg_num + 2;
    py::array_t<int> S_array = py::array_t<int>({user_num * perUserNum, row});
    py::buffer_info buf_S = S_array.request();
    int *ptr = (int *)buf_S.ptr;

    for (int user = 0; user < user_num; user++)
    {
        std::vector<int> pos_item = allPos[user];

        for (int pair_i = 0; pair_i < perUserNum; pair_i++)
        {
            int negitem = 0;
            ptr[(user * perUserNum + pair_i) * row] = user;
            ptr[(user * perUserNum + pair_i) * row + 1] = pos_item[randint_(pos_item.size())];
            for (int index = 2; index < neg_num + 2; index++)
            {
                do
                {
                    negitem = randint_(item_num);
                } while (
                    find(pos_item.begin(), pos_item.end(), negitem) != pos_item.end());
                ptr[(user * perUserNum + pair_i) * row + index] = negitem;
            }
        }
    }
    return S_array;
}

py::array_t<int> sample_negative_ByUser(std::vector<int> users, int item_num, std::vector<std::vector<int>> allPos, int neg_num)
{
    int row = neg_num + 2;
    int col = users.size();
    py::array_t<int> S_array = py::array_t<int>({col, row});
    py::buffer_info buf_S = S_array.request();
    int *ptr = (int *)buf_S.ptr;

    for (int user_i = 0; user_i < users.size(); user_i++)
    {
        int user = users[user_i];
        std::vector<int> pos_item = allPos[user];
        int negitem = 0;

        ptr[user_i * row] = user;
        ptr[user_i * row + 1] = pos_item[randint_(pos_item.size())];

        for (int neg_i = 2; neg_i < row; neg_i++)
        {
            do
            {
                negitem = randint_(item_num);
            } while (
                find(pos_item.begin(), pos_item.end(), negitem) != pos_item.end());
            ptr[user_i * row + neg_i] = negitem;
        }
    }
    return S_array;
}

void set_seed(unsigned int seed)
{
    srand(seed);
}

using namespace py::literals;

PYBIND11_MODULE(sampling, m)
{
    srand(time(0));
    // srand(2020);
    m.doc() = "example plugin";
    m.def("randint", &randint_, "generate int between [0 end]", "end"_a);
    m.def("seed", &set_seed, "set random seed", "seed"_a);
    m.def("sample_negative", &sample_negative, "sampling negatives for all",
          "user_num"_a, "item_num"_a, "train_num"_a, "allPos"_a, "neg_num"_a);
    m.def("sample_negative_ByUser", &sample_negative_ByUser, "sampling negatives for given users",
          "users"_a, "item_num"_a, "allPos"_a, "neg_num"_a);
}