#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void WeightedEuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  CHECK_EQ(bottom[0]->count(1), bottom[2]->count(1)) << "Weight data must have same dimension";
  diff_.ReshapeLike(*bottom[0]);
  weighted_squared_diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void WeightedEuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  //LOG
  // diff_ = a - b
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());

  // weighted_squared_diff_ = diff.^2
  caffe_mul(
      count,
      diff_.cpu_data(),
      diff_.cpu_data(),
      weighted_squared_diff_.mutable_cpu_data()
  );

  // weighted_squared_diff_ = weighted_squared_diff_ .* weights
  Apply_weights( weighted_squared_diff_, *bottom[2] );
  // Since we're keeping diff_ around for the purposes of 
  // gradients, apply the weights again
  Apply_weights( diff_, *bottom[2] );

  //printf("num %d count %d\n", bottom[2]->count(), bottom[2]->num() );
  Dtype loss = caffe_cpu_asum( count, weighted_squared_diff_.cpu_data() );
  loss = loss / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void WeightedEuclideanLossLayer<Dtype>::Apply_weights( Blob<Dtype>& blob,
    Blob<Dtype>& bottomWeights ) {
  size_t d = (size_t)(blob.count() / blob.num());
  const Dtype* weights;
  for (size_t idx = 0; idx < blob.num(); idx++) {
      // if we have weights per example
      if( bottomWeights.num() > 1 ) {
        weights = bottomWeights.cpu_data()+idx*d;
      } else { // if we have one set of weights total
        weights = bottomWeights.cpu_data();
      }
      caffe_mul(
          d,
          blob.mutable_cpu_data()+idx*d,
          weights,
          blob.mutable_cpu_data()+idx*d
      );
      Dtype sumWeights = caffe_cpu_asum( d, weights );
      Dtype weightScale = Dtype(d) / sumWeights;
      caffe_scal(
          d,
          weightScale,
          blob.mutable_cpu_data()+idx*d
      );
  }
}
      /*
      TODO: do we have to normalize the weights?
      Dtype sumweights = caffe_cpu_asum<Dtype>(
          d, weights
      );
      printf("Sumweights %.2f\n", sumweights);
      caffe_cpu_scale(
          d,
          Dtype(1.0 / sumweights),
          diff_.mutable_cpu_data() + idx*d,
          diff_.mutable_cpu_data() + idx*d
      );
      */

template <typename Dtype>
void WeightedEuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(WeightedEuclideanLossLayer);
#endif

INSTANTIATE_CLASS(WeightedEuclideanLossLayer);
REGISTER_LAYER_CLASS(WeightedEuclideanLoss);

}  // namespace caffe
