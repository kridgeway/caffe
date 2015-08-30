#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <iomanip>

namespace caffe {

template <typename Dtype> 
void StatsLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, 
  const vector<Blob<Dtype>*>& top) {
  output_filename_ = this->layer_param_.stats_param().output_file();
  ofs_ = new std::ofstream( output_filename_.c_str(), std::ofstream::out );
  precision_ = this->layer_param_.stats_param().precision();
}

template <typename Dtype>
void StatsLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, 
  const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void StatsLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  for (int i = 0; i < num; ++i) {
    std::vector<std::pair<Dtype, int> > bottom_data_vector;
    for (int j = 0; j < dim; ++j) {
      bottom_data_vector.push_back(
          std::make_pair(bottom_data[i * dim + j], j));
    }
    (*ofs_) << std::setprecision(precision_);
    for (int k = 0; k < dim; k++) {
        (*ofs_) << (double)bottom_data[i*dim+k] << " ";
    }
    (*ofs_) << std::endl;
  }
}

INSTANTIATE_CLASS(StatsLayer);
REGISTER_LAYER_CLASS(Stats);

}  // namespace caffe
