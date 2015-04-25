#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {
template <typename Dtype>
void NonBinaryPenaltyLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  non_binary_penalty_.ReshapeLike(*bottom[0]);
}
}

template <typename Dtype>
void NonBinaryPenaltyLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int count = bottom[0]->count();
  // nbp_ = x^2
  caffe_powx(
      count,
      bottom[0]->cpu_data(),
      Dtype(2),
      non_binary_penalty_.mutable_cpu_data()
  );

  // nbp_ = -1 * x^2
  caffe_cpu_axpby(
      count,
      Dtype(-1.0),
      non_binary_penalty_.cpu_data(),
      Dtype(0.0),
      non_binary_penalty_.mutable_cpu_data()
  );

  // nbp_ = -1 * x^2 + 1
  caffe_add_scalar(
      count,
      Dtype(1.0),
      non_binary_penalty_.mutable_cpu_data()
  );

  Dtype loss = caffe_cpu_asum(
      count,
      non_binary_penalty_.cpu_data() );
  loss = loss / bottom[0]->num();
  top[0]->mutable_cpu_data()[0] = loss;
}


template <typename Dtype>
void NonBinaryPenaltyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if( propagate_down[0] ) {
      caffe_cpu_axpby(
        bottom[0]->count(),
        Dtype(-2.0),
        bottom[0]->cpu_data(),
        Dtype(0),
        bottom[0]->mutable_cpu_diff()
      );
    }
}

#ifdef CPU_ONLY
STUB_GPU(NonBinaryPenaltyLossLayer);
#endif

INSTANTIATE_CLASS(NonBinaryPenaltyLossLayer);
REGISTER_LAYER_CLASS(NonBinaryPenaltyLoss);
