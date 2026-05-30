package com.datalakeattendance.faceprocessor

import com.facebook.react.ReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.uimanager.ViewManager

/**
 * FaceAnalyzerPackage — registers FaceAnalyzerModule with React Native.
 *
 * Register in MainApplication.kt:
 *   override fun getPackages(): List<ReactPackage> =
 *       PackageList(this).packages + listOf(FaceAnalyzerPackage())
 */
class FaceAnalyzerPackage : ReactPackage {

    override fun createNativeModules(
        reactContext: ReactApplicationContext
    ): List<NativeModule> = listOf(FaceAnalyzerModule(reactContext))

    override fun createViewManagers(
        reactContext: ReactApplicationContext
    ): List<ViewManager<*, *>> = emptyList()
}
