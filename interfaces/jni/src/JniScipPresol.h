/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class de_zib_jscip_nativ_jni_JniScipPresol */

#ifndef _Included_de_zib_jscip_nativ_jni_JniScipPresol
#define _Included_de_zib_jscip_nativ_jni_JniScipPresol
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     de_zib_jscip_nativ_jni_JniScipPresol
 * Method:    presolGetData
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_de_zib_jscip_nativ_jni_JniScipPresol_presolGetData
  (JNIEnv *, jobject, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipPresol
 * Method:    presolSetData
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_de_zib_jscip_nativ_jni_JniScipPresol_presolSetData
  (JNIEnv *, jobject, jlong, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipPresol
 * Method:    presolGetName
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_de_zib_jscip_nativ_jni_JniScipPresol_presolGetName
  (JNIEnv *, jobject, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipPresol
 * Method:    presolGetDesc
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_de_zib_jscip_nativ_jni_JniScipPresol_presolGetDesc
  (JNIEnv *, jobject, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipPresol
 * Method:    presolGetPriority
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_de_zib_jscip_nativ_jni_JniScipPresol_presolGetPriority
  (JNIEnv *, jobject, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipPresol
 * Method:    presolIsDelayed
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_de_zib_jscip_nativ_jni_JniScipPresol_presolIsDelayed
  (JNIEnv *, jobject, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipPresol
 * Method:    presolWasDelayed
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_de_zib_jscip_nativ_jni_JniScipPresol_presolWasDelayed
  (JNIEnv *, jobject, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipPresol
 * Method:    presolIsInitialized
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_de_zib_jscip_nativ_jni_JniScipPresol_presolIsInitialized
  (JNIEnv *, jobject, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipPresol
 * Method:    presolGetSetupTime
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_de_zib_jscip_nativ_jni_JniScipPresol_presolGetSetupTime
  (JNIEnv *, jobject, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipPresol
 * Method:    presolGetTime
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_de_zib_jscip_nativ_jni_JniScipPresol_presolGetTime
  (JNIEnv *, jobject, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipPresol
 * Method:    presolGetNFixedVars
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_de_zib_jscip_nativ_jni_JniScipPresol_presolGetNFixedVars
  (JNIEnv *, jobject, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipPresol
 * Method:    presolGetNAggrVars
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_de_zib_jscip_nativ_jni_JniScipPresol_presolGetNAggrVars
  (JNIEnv *, jobject, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipPresol
 * Method:    presolGetNChgVarTypes
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_de_zib_jscip_nativ_jni_JniScipPresol_presolGetNChgVarTypes
  (JNIEnv *, jobject, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipPresol
 * Method:    presolGetNChgBds
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_de_zib_jscip_nativ_jni_JniScipPresol_presolGetNChgBds
  (JNIEnv *, jobject, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipPresol
 * Method:    presolGetNAddHoles
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_de_zib_jscip_nativ_jni_JniScipPresol_presolGetNAddHoles
  (JNIEnv *, jobject, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipPresol
 * Method:    presolGetNDelConss
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_de_zib_jscip_nativ_jni_JniScipPresol_presolGetNDelConss
  (JNIEnv *, jobject, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipPresol
 * Method:    presolGetNAddConss
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_de_zib_jscip_nativ_jni_JniScipPresol_presolGetNAddConss
  (JNIEnv *, jobject, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipPresol
 * Method:    presolGetNUpgdConss
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_de_zib_jscip_nativ_jni_JniScipPresol_presolGetNUpgdConss
  (JNIEnv *, jobject, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipPresol
 * Method:    presolGetNChgCoefs
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_de_zib_jscip_nativ_jni_JniScipPresol_presolGetNChgCoefs
  (JNIEnv *, jobject, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipPresol
 * Method:    presolGetNChgSides
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_de_zib_jscip_nativ_jni_JniScipPresol_presolGetNChgSides
  (JNIEnv *, jobject, jlong);

/*
 * Class:     de_zib_jscip_nativ_jni_JniScipPresol
 * Method:    presolGetNCalls
 * Signature: (J)I
 */
JNIEXPORT jint JNICALL Java_de_zib_jscip_nativ_jni_JniScipPresol_presolGetNCalls
  (JNIEnv *, jobject, jlong);

#ifdef __cplusplus
}
#endif
#endif
