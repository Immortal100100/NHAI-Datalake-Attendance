#import <React/RCTBridgeModule.h>

/**
 * Objective-C bridge macro — exposes FaceAnalyzerModule.swift to React Native's JS runtime.
 * No implementation needed here; Swift handles everything in FaceAnalyzerModule.swift.
 */
@interface RCT_EXTERN_MODULE(FaceAnalyzerModule, NSObject)

RCT_EXTERN_METHOD(
  loadModels: (RCTPromiseResolveBlock)resolve
  reject: (RCTPromiseRejectBlock)reject
)

RCT_EXTERN_METHOD(
  analyzeFrame: (nonnull NSNumber *)width
  height: (nonnull NSNumber *)height
  pixelData: (NSString *)pixelData
  runFaceNet: (BOOL)runFaceNet
  resolve: (RCTPromiseResolveBlock)resolve
  reject: (RCTPromiseRejectBlock)reject
)

RCT_EXTERN_METHOD(
  releaseModels: (RCTPromiseResolveBlock)resolve
  reject: (RCTPromiseRejectBlock)reject
)

@end
