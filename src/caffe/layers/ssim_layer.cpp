#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <cv.h>	

namespace caffe {

template <typename Dtype>
void SSIMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //int x=img1_temp->width, y=img1_temp->height;
  size_t dim = (size_t)(bottom[0]->count() / bottom[0]->num());
  //TODO this assumes square images with 3 channels, which might not be the case
  int x = (int)floor( sqrt(dim/3) );
  int y = x;
	int nChan=3, d=IPL_DEPTH_32F;
	CvSize size = cvSize(x, y);

  img1_temp = cvCreateImage( size, d, nChan);
  img2_temp = cvCreateImage( size, d, nChan);
	img1 = cvCreateImage( size, d, nChan);
	img2 = cvCreateImage( size, d, nChan);
	img1_sq = cvCreateImage( size, d, nChan);
	img2_sq = cvCreateImage( size, d, nChan);
	img1_img2 = cvCreateImage( size, d, nChan);
	mu1 = cvCreateImage( size, d, nChan);
	mu2 = cvCreateImage( size, d, nChan);
	mu1_sq = cvCreateImage( size, d, nChan);
	mu2_sq = cvCreateImage( size, d, nChan);
	mu1_mu2 = cvCreateImage( size, d, nChan);
	sigma1_sq = cvCreateImage( size, d, nChan);
	sigma2_sq = cvCreateImage( size, d, nChan);
	sigma12 = cvCreateImage( size, d, nChan);
	temp1 = cvCreateImage( size, d, nChan);
	temp2 = cvCreateImage( size, d, nChan);
	temp3 = cvCreateImage( size, d, nChan);
	ssim_map = cvCreateImage( size, d, nChan);
}

template <typename Dtype>
SSIMLayer<Dtype>::~SSIMLayer() {
  cvReleaseImage(&img1);
  cvReleaseImage(&img2);
  cvReleaseImage(&img1_temp);
  cvReleaseImage(&img2_temp);
  cvReleaseImage(&img1_sq);
  cvReleaseImage(&img2_sq);
  cvReleaseImage(&mu1);
  cvReleaseImage(&mu2);
  cvReleaseImage(&mu1_sq);
  cvReleaseImage(&mu2_sq);
  cvReleaseImage(&mu1_mu2);
  cvReleaseImage(&sigma1_sq);
  cvReleaseImage(&sigma2_sq);
  cvReleaseImage(&sigma12);
  cvReleaseImage(&ssim_map);
  cvReleaseImage(&temp1);
  cvReleaseImage(&temp2);
  cvReleaseImage(&temp3);
}

template <typename Dtype>
void SSIMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                  const vector<Blob<Dtype>*>& top) {
  for (int i = 1; i < bottom.size(); ++i) {
    CHECK(bottom[i]->shape() == bottom[0]->shape());
  }
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void SSIMLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  size_t dim = (size_t)(bottom[0]->count() / bottom[0]->num());
  int width = (int)( sqrt(dim/3) );
  int height = width;
  int nChan=3, d=IPL_DEPTH_32F;
  // default settings
  double C1 = 6.5025, C2 = 58.5225;
  for( size_t image_idx = 0; image_idx < bottom[0]->num(); image_idx++ ) {
    // TODO step 1, get image data into img1 and img2
    cvSetData(img1, (void*)bottom[0]->cpu_data(), (int)sizeof(Dtype)*width*nChan );
    cvSetData(img2, (void*)bottom[1]->cpu_data(), (int)sizeof(Dtype)*width*nChan );

    // Step 2, calculate ssim_map
    //cvConvert(img1_temp, img1);
    //cvConvert(img2_temp, img2);
    //cvReleaseImage(&img1_temp);
    //cvReleaseImage(&img2_temp);
    
    cvPow( img1, img1_sq, 2 );
    cvPow( img2, img2_sq, 2 );
    cvMul( img1, img2, img1_img2, 1 );

    cvSmooth( img1, mu1, CV_GAUSSIAN, 11, 11, 1.5 );
    cvSmooth( img2, mu2, CV_GAUSSIAN, 11, 11, 1.5 );
    
    cvPow( mu1, mu1_sq, 2 );
    cvPow( mu2, mu2_sq, 2 );
    cvMul( mu1, mu2, mu1_mu2, 1 );

    cvSmooth( img1_sq, sigma1_sq, CV_GAUSSIAN, 11, 11, 1.5 );
    cvAddWeighted( sigma1_sq, 1, mu1_sq, -1, 0, sigma1_sq );
    
    cvSmooth( img2_sq, sigma2_sq, CV_GAUSSIAN, 11, 11, 1.5 );
    cvAddWeighted( sigma2_sq, 1, mu2_sq, -1, 0, sigma2_sq );

    cvSmooth( img1_img2, sigma12, CV_GAUSSIAN, 11, 11, 1.5 );
    cvAddWeighted( sigma12, 1, mu1_mu2, -1, 0, sigma12 );

    cvScale( mu1_mu2, temp1, 2 );
    cvAddS( temp1, cvScalarAll(C1), temp1 );

    // (2*sigma12 + C2)
    cvScale( sigma12, temp2, 2 );
    cvAddS( temp2, cvScalarAll(C2), temp2 );

    // ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    cvMul( temp1, temp2, temp3, 1 );

    // (mu1_sq + mu2_sq + C1)
    cvAdd( mu1_sq, mu2_sq, temp1 );
    cvAddS( temp1, cvScalarAll(C1), temp1 );

    // (sigma1_sq + sigma2_sq + C2)
    cvAdd( sigma1_sq, sigma2_sq, temp2 );
    cvAddS( temp2, cvScalarAll(C2), temp2 );

    // ((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2))
    cvMul( temp1, temp2, temp1, 1 );

    // ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2))
    cvDiv( temp3, temp1, ssim_map, 1 );

    // Step 3 copy ssim_map to top
    Dtype* data = (Dtype*)ssim_map->imageData;
    memcpy( top[0]->mutable_cpu_data(), data, sizeof(Dtype)*ssim_map->width*ssim_map->height*ssim_map->nChannels );
  }
}

}
