#include "core/cuda/kernels/prepared_fastdvdnet_kernels.h"

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace studiocast::cuda;
using namespace studiocast::cuda::kernels;

int main() {
  CudaDriverApi driver;
  std::string error;
  if (!driver.Initialize(&error) || !driver.EnsureContextForDevice(0, &error)) {
    std::cout << "SKIP: " << error << '\n';
    return 77;
  }
  CUstream stream=nullptr;
  if (!driver.CreateStream(&stream,&error)) { std::cerr<<error<<'\n'; return 1; }
  CUcontext context=nullptr;
  if (driver.f().cuCtxGetCurrent(&context) != CUDA_SUCCESS || !context) return 1;
  const auto cleanup_stream=[&]{driver.DestroyStream(stream,nullptr);};
  CudaImage input, output;
  CudaTensor oldest,middle,current,noisy,noise,denoised;
  PreparedFastDvdnetKernels kernels;
  bool ok=input.Allocate(&driver,3,2,PixelFormatGpu::rgb_u8,&error)&&
      output.Allocate(&driver,3,2,PixelFormatGpu::rgb_u8,&error)&&
      oldest.AllocateNchwF32(&driver,1,3,4,4,&error)&&
      middle.AllocateNchwF32(&driver,1,3,4,4,&error)&&
      current.AllocateNchwF32(&driver,1,3,4,4,&error)&&
      noisy.AllocateNchwF32(&driver,1,15,4,4,&error)&&
      noise.AllocateNchwF32(&driver,1,1,4,4,&error)&&
      denoised.AllocateNchwF32(&driver,1,3,4,4,&error)&&
      PrepareFastDvdnetKernels(driver,context,stream,&kernels,&error);
  if(!ok){std::cerr<<error<<'\n';cleanup_stream();return 1;}
  const std::uint8_t rgb[]={0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170};
  std::vector<float> old(48,0.25f), mid(48,0.5f), den(48);
  for(std::size_t i=0;i<den.size();++i) den[i]=static_cast<float>(i/16)*0.25f+static_cast<float>(i%16)/30.0f;
  ok=input.UploadFromCpuRgb24(&driver,rgb,9,stream,&error)&&oldest.UploadFromCpuF32(&driver,old.data(),old.size(),stream,&error)&&middle.UploadFromCpuF32(&driver,mid.data(),mid.size(),stream,&error)&&denoised.UploadFromCpuF32(&driver,den.data(),den.size(),stream,&error)&&
    LaunchFastDvdnetPreprocess(kernels,input,oldest,middle,current,noisy,noise,false,15.0f/255.0f,&error)&&
    LaunchFastDvdnetPostprocess(kernels,denoised,output,&error)&&driver.StreamSynchronize(stream,&error);
  std::vector<float> got_noisy,got_noise;std::vector<std::uint8_t> got_rgb(18);
  ok=ok&&noisy.DownloadToCpuF32(&driver,&got_noisy,stream,&error)&&noise.DownloadToCpuF32(&driver,&got_noise,stream,&error)&&output.DownloadToCpuRgb24(&driver,got_rgb.data(),9,stream,&error)&&driver.StreamSynchronize(stream,&error);
  auto near=[](float a,float b){return std::fabs(a-b)<=1e-6f;};
  ok=ok&&got_noisy.size()==240&&got_noise.size()==16;
  for(int i=0;ok&&i<16;++i){ok=near(got_noisy[i],0.25f)&&near(got_noisy[3*16+i],0.5f)&&near(got_noise[i],15.0f/255.0f);}
  // Bottom/right padding replicates source pixel (2,1): RGB 150,160,170.
  ok=ok&&near(got_noisy[6*16+15],150.0f/255.0f)&&near(got_noisy[7*16+15],160.0f/255.0f)&&near(got_noisy[8*16+15],170.0f/255.0f);
  for(int y=0;ok&&y<2;++y)for(int x=0;ok&&x<3;++x)for(int c=0;ok&&c<3;++c){const auto expected=static_cast<std::uint8_t>(std::nearbyint(std::clamp(den[c*16+y*4+x],0.0f,1.0f)*255.0f));ok=got_rgb[y*9+x*3+c]==expected;}
  if(!ok)std::cerr<<"FastDVDnet generated-frame reference mismatch: geometry=3x2 padded=4x4 tolerance=1e-6\n";
  DestroyFastDvdnetKernels(&kernels,nullptr);denoised.Free(&driver,nullptr);noise.Free(&driver,nullptr);noisy.Free(&driver,nullptr);current.Free(&driver,nullptr);middle.Free(&driver,nullptr);oldest.Free(&driver,nullptr);output.Free(&driver,nullptr);input.Free(&driver,nullptr);cleanup_stream();
  if(ok)std::cout<<"PASS: device=0 geometry=3x2 padded=4x4 uploads=4 downloads=3 syncs=2 tolerance=1e-6\n";
  return ok?0:1;
}
