#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/loss_layers.hpp"

namespace caffe {
template <typename Dtype>
void NonBinaryPenaltyLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //the super call assumes there are two bottoms for some reason,
  //so don't call it
  //LossLayer<Dtype>::Reshape(bottom, top);
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
  non_binary_penalty_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void NonBinaryPenaltyLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  int count = bottom[0]->count();
  // nbp_ = x^2
  caffe_powx(
      count,
      bottom[0]->cpu_data(),
      Dtype(2.0),
      non_binary_penalty_.mutable_cpu_data()
  );

  // nbp_ = -1 * x^2
  caffe_scal(
      count,
      Dtype(-1.0),
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
      // 2*x?
      Dtype alpha = Dtype(-1.0) * top[0]->cpu_diff()[0] / bottom[0]->num();
      caffe_cpu_axpby(
        bottom[0]->count(),
        alpha,
        bottom[0]->cpu_data(),
        Dtype(0),
        bottom[0]->mutable_cpu_diff()
      );
      /*
       * 1.0000 128 -1.00000000 -0.01562500
       * 1.0000 128 1.00000000 0.01562500
       * 1.0000 128 0.99999911 0.01562499
       * 1.0000 128 -1.00000000 -0.01562500
       * 1.0000 128 -1.00000000 -0.01562500*/
      /*printf("%.4f %d %.8f %.8f\n", top[0]->cpu_diff()[0],
          bottom[0]->num(),
          bottom[0]->cpu_data()[0], bottom[0]->cpu_diff()[0] );
          */
    }
}

#ifdef CPU_ONLY
STUB_GPU(NonBinaryPenaltyLossLayer);
#endif

INSTANTIATE_CLASS(NonBinaryPenaltyLossLayer);
REGISTER_LAYER_CLASS(NonBinaryPenaltyLoss);

}
