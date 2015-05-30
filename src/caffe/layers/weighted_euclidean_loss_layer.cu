#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
/*

namespace caffe {
template <typename Dtype>
void WeightedEuclideanLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  // diff_ = a - b
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());

  // weighted_squared_diff_ = diff.^2
  caffe_gpu_mul(
      count,
      diff_.gpu_data(),
      diff_.gpu_data(),
      weighted_squared_diff_.mutable_gpu_data()
  );

  Apply_weights_gpu( weighted_squared_diff_, *bottom[2] );

  Apply_weights( diff_, *bottom[2] );

  Dtype loss = caffe_gpu_asum( count, 
      weighted_squared_diff_.gpu_data() );

  loss = loss / bottom[0]->num() / Dtype(2);

  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void WeightedEuclideanLossLayer<Dtype>::Apply_weights_gpu( Blob<Dtype>& blob,
    Blob<Dtype>& bottomWeights ) {
  size_t d = (size_t)(blob.count() / blob.num());
  const Dtype* weights;
  for (size_t idx = 0; idx < blob.num(); idx++) {
      // if we have weights per example
      if( bottomWeights.num() > 1 ) {
        weights = bottomWeights.mutable_gpu_data()+idx*d;
      } else { // if we have one set of weights total
        weights = bottomWeights.mutable_gpu_data();
      }
      caffe_mul(
          d,
          blob.mutable_cpu_data()+idx*d,
          weights,
          blob.mutable_cpu_data()+idx*d
      );
  }
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}
*/

